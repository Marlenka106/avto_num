"""Оценка YOLOv8m на аугментированном датасете."""

from pathlib import Path

from ultralytics import YOLO

# fmt: off
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'D', 'E', 'G', 'H', 'J', 'K', 'L', 'N',
           'R', 'S', 'T', 'U', 'V', 'X', 'Z']
# fmt: on


def get_gt_from_txt(txt_path):
    if not txt_path.exists():
        return ""
    boxes_chars = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, cx = int(parts[0]), float(parts[1])
                boxes_chars.append((cx, CLASSES[class_id]))
    boxes_chars.sort(key=lambda x: x[0])
    return "".join([c for _, c in boxes_chars])


def filter_plate_predictions(preds):
    if len(preds) == 0:
        return []
    preds.sort(key=lambda x: x[0])
    heights = [p[2][3] - p[2][1] for p in preds]
    y_centers = [(p[2][1] + p[2][3]) / 2 for p in preds]

    med_h = sorted(heights)[len(heights) // 2]
    med_y = sorted(y_centers)[len(y_centers) // 2]

    filtered = []
    for cx, char, box, conf in preds:
        h = box[3] - box[1]
        y = (box[1] + box[3]) / 2
        # Фильтр: высота и линия должны совпадать с медианой
        if 0.7 * med_h < h < 1.3 * med_h and abs(y - med_y) < 0.4 * med_h:
            filtered.append((cx, char, conf))

    # Защита от галлюцинаций: максимум 7 символов
    if len(filtered) > 7:
        filtered.sort(key=lambda x: x[2], reverse=True)
        filtered = filtered[:7]
        filtered.sort(key=lambda x: x[0])

    return [p[1] for p in filtered]


def evaluate():
    # Путь к новой модели
    model_path = "runs/detect/runs/yolo_aug/plates_detector_m/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Модель не найдена: {model_path}")
        return

    model = YOLO(model_path)
    test_dir = Path("yolo_dataset_augmented/images/test")
    labels_dir = Path("yolo_dataset_augmented/labels/test")

    correct = 0
    total = 0
    CONF_THRESHOLD = 0.25  # Начнем с 0.25, новая модель лучше калибрована

    print(f"\n🔍 Оценка YOLOv8m на {len(list(test_dir.glob('*.jpeg')))} изображениях...\n")

    for img_path in sorted(test_dir.glob("*.jpeg")):
        stem = img_path.stem
        txt_path = labels_dir / f"{stem}.txt"
        gt_str = get_gt_from_txt(txt_path)

        results = model(img_path, conf=CONF_THRESHOLD, iou=0.45, verbose=False)

        preds = []
        if len(results[0].boxes) > 0:
            for box, cls_id, conf in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.cls.cpu().numpy(),
                results[0].boxes.conf.cpu().numpy(),
                strict=False,
            ):
                cx = (box[0] + box[2]) / 2
                preds.append((cx, CLASSES[int(cls_id)], box, conf))

        pred_chars = filter_plate_predictions(preds)
        pred_str = "".join(pred_chars)

        if pred_str == gt_str and len(gt_str) > 0:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        total += 1
        print(f"[{total:2d}/20] {status} GT: {gt_str: <10} | Pred: {pred_str: <10}")

    accuracy = correct / total
    print("\n" + "=" * 55)
    print(f"🎯 EXACT MATCH ACCURACY: {accuracy:.2%} ({correct}/{total})")
    print("=" * 55)

    if accuracy >= 0.80:
        print("🏆 ЦЕЛЬ ДОСТИГНУТА: Точность >= 80%! 🎉")


if __name__ == "__main__":
    evaluate()
