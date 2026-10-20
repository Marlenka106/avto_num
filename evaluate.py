"""Оценка детектора (YOLO или FCOS): Exact Match по номерам, CER и 95% доверительный интервал.

Порог уверенности подбирается на val (можно передать несколько значений --conf),
на test запускается один раз с выбранным порогом.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from src.data.classes import CLASSES, read_plate
from src.data.dataset import get_transform
from src.data.images import list_images
from src.data.quality import is_clean_annotation
from src.evaluation.inference import decode_predictions, to_detections
from src.evaluation.metrics import cer, wilson_interval
from src.evaluation.postprocess import filter_plate_predictions, sort_left_to_right
from src.models.fcos import FCOSDetector, compute_locations


def make_yolo_predictor(model, conf):
    """predict(img_path) -> список (box [x1, y1, x2, y2], class_id, conf)."""

    def predict(img_path):
        # NMS без учёта класса: иначе две гипотезы одного символа (например, B и 8) остаются обе
        boxes = model(img_path, conf=conf, iou=0.45, agnostic_nms=True, verbose=False)[0].boxes
        return list(zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), strict=True))

    return predict


def make_fcos_predictor(model, conf):
    """predict(img_path) -> список (box [x1, y1, x2, y2], class_id, conf) в координатах входа 256x512."""
    device = next(model.parameters()).device
    transform = get_transform(train=False)

    def predict(img_path):
        image = np.array(Image.open(img_path).convert('RGB'))
        tensor = transform(image=image, bboxes=[], labels=[])['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            logits, bbox_reg, centerness = model(tensor)
        # Геометрический фильтр общий для обеих моделей - применяется в evaluate()
        pred = decode_predictions(
            logits,
            bbox_reg,
            centerness,
            compute_locations(logits),
            conf_thresh=conf,
            nms_thresh=0.45,
            plate_filter=False,
        )[0]
        return to_detections(pred)

    return predict


def load_fcos(weights):
    """FCOS с весами из weights/best.pt."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCOSDetector(num_classes=len(CLASSES), backbone_pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.eval()


def evaluate(img_paths, labels_dir, predict, train_plates, postprocess=filter_plate_predictions):
    """Сравнивает предсказанные номера с разметкой."""
    rows = []
    for img_path in img_paths:
        gt = read_plate(labels_dir / f"{img_path.stem}.txt")
        pred = "".join(CLASSES[class_id] for class_id in postprocess(predict(img_path)))
        rows.append({"name": img_path.name, "gt": gt, "pred": pred, "seen": gt in train_plates})
    return rows


def summarize(rows, title):
    """Печатает Exact Match с доверительным интервалом, средний CER и Exact Match по номерам."""
    correct = sum(row["gt"] == row["pred"] for row in rows)
    low, high = wilson_interval(correct, len(rows))
    mean_cer = sum(cer(row["gt"], row["pred"]) for row in rows) / len(rows)

    # Один номер снят многократно: среднее по номерам не даёт частому номеру перевесить остальные
    plates = defaultdict(list)
    for row in rows:
        plates[row["gt"]].append(row["gt"] == row["pred"])
    per_plate = sum(sum(hits) / len(hits) for hits in plates.values()) / len(plates)

    print(
        f"{title}: Exact Match {correct / len(rows):.1%} ({correct}/{len(rows)}), "
        f"95% ДИ [{low:.1%}; {high:.1%}], CER {mean_cer:.1%}, по номерам {per_plate:.1%} ({len(plates)} номеров)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["yolo", "fcos"], default="yolo")
    parser.add_argument("--weights", required=True, help="путь к best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25])
    parser.add_argument("--data", type=Path, default=Path("yolo_dataset"))
    parser.add_argument("--no-filter", action="store_true", help="без геометрического фильтра - для сравнения с ним")
    args = parser.parse_args()
    postprocess = sort_left_to_right if args.no_filter else filter_plate_predictions

    images_dir = args.data / "images" / args.split
    labels_dir = args.data / "labels" / args.split
    all_images = list_images(images_dir)
    # Неполная разметка делает правильный ответ модели «ошибкой» - такие изображения не оцениваем
    images = [p for p in all_images if is_clean_annotation(labels_dir / f"{p.stem}.txt")]
    train_plates = {read_plate(p) for p in (args.data / "labels" / "train").glob("*.txt")}
    print(f"{args.split}: {len(images)} изображений, исключено с неполной разметкой: {len(all_images) - len(images)}\n")

    if args.model == "yolo":
        model, make_predictor = YOLO(args.weights), make_yolo_predictor
    else:
        model, make_predictor = load_fcos(args.weights), make_fcos_predictor

    for conf in args.conf:
        rows = evaluate(images, labels_dir, make_predictor(model, conf), train_plates, postprocess)
        if len(args.conf) == 1:
            for row in rows:
                status = "✅" if row["gt"] == row["pred"] else "❌"
                print(f"{status} {row['name']:<32} GT: {row['gt']:<8} | Pred: {row['pred']}")
            print()

        summarize(rows, f"conf={conf}")
        # В официальном test многие номера есть и в train: обобщение видно только на «новых» номерах
        for title, subset in [("  номер есть в train", True), ("  новый номер", False)]:
            subset_rows = [row for row in rows if row["seen"] == subset]
            if subset_rows:
                summarize(subset_rows, title)


if __name__ == "__main__":
    main()
