"""Оценка кастомной модели FCOS на аугментированном датасете."""

from pathlib import Path

import torch
import torchvision.ops as ops
from loguru import logger
from torch.utils.data import DataLoader

from src.data.dataset import CLASSES, PlateCharDataset, get_transform
from src.models.fcos import FCOSDetector

INV_CLASSES = dict(enumerate(CLASSES))


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def generate_locations(h, w, device):
    strides = [8, 16, 32]
    locations = []
    for stride in strides:
        feat_h, feat_w = h // stride, w // stride
        shifts_x = torch.arange(0, feat_w * stride, stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, feat_h * stride, stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        loc = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)
        locations.append(loc)
    return locations


def decode_predictions(logits, bbox_reg, centerness, locations, conf_thresh=0.25, nms_thresh=0.45):
    batch_size = logits[0].shape[0]
    num_classes = logits[0].shape[1]
    all_predictions = []

    for b in range(batch_size):
        all_boxes, all_scores, all_labels = [], [], []
        for level in range(len(locations)):
            loc = locations[level]
            log = logits[level][b].permute(1, 2, 0).reshape(-1, num_classes)
            reg = bbox_reg[level][b].permute(1, 2, 0).reshape(-1, 4)
            cent = centerness[level][b].permute(1, 2, 0).reshape(-1)

            scores_lvl = torch.sigmoid(log) * torch.sigmoid(cent).unsqueeze(1)
            pos = torch.where(scores_lvl > conf_thresh)
            if pos[0].numel() == 0:
                continue

            idx_pos, idx_cls = pos[0], pos[1]
            r = reg[idx_pos]
            l = loc[idx_pos]
            bx = torch.stack([l[:, 0] - r[:, 0], l[:, 1] - r[:, 1], l[:, 0] + r[:, 2], l[:, 1] + r[:, 3]], dim=1)

            all_boxes.append(bx)
            all_scores.append(scores_lvl[idx_pos, idx_cls])
            all_labels.append(idx_cls)

        if len(all_boxes) == 0:
            all_predictions.append(
                {
                    'boxes': torch.zeros((0, 4)),
                    'scores': torch.zeros((0,)),
                    'labels': torch.zeros((0,), dtype=torch.long),
                }
            )
            continue

        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)

        # Class-agnostic NMS
        keep = ops.nms(boxes, scores, nms_thresh)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        order = boxes[:, 0].argsort()
        all_predictions.append({'boxes': boxes[order], 'scores': scores[order], 'labels': labels[order]})

    return all_predictions


def filter_plate_predictions(preds):
    """Тот же умный фильтр, что и в YOLO"""
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
        if 0.7 * med_h < h < 1.3 * med_h and abs(y - med_y) < 0.4 * med_h:
            filtered.append((cx, char, conf))

    if len(filtered) > 7:
        filtered.sort(key=lambda x: x[2], reverse=True)
        filtered = filtered[:7]
        filtered.sort(key=lambda x: x[0])

    return [p[1] for p in filtered]


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


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. Загружаем тестовый набор
    test_dataset = PlateCharDataset(
        img_dir="yolo_dataset_augmented/images/test",
        txt_dir="yolo_dataset_augmented/labels/test",
        transforms=get_transform(train=False),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 2. Загружаем модель FCOS
    model = FCOSDetector(num_classes=27, backbone_pretrained=False).to(device)
    weights_path = Path("weights/best.pt")

    if not weights_path.exists():
        logger.error(f"FCOS weights not found at {weights_path}. Train first!")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Loaded FCOS weights from {weights_path}")

    # 3. Инференс
    # ВАЖНО: 256, 512, так как get_transform делает Resize((256, 512))
    locations = generate_locations(256, 512, device)

    correct = 0
    total = len(test_dataset)
    CONF_THRESH = 0.25

    logger.info(f"Running FCOS inference on {total} test images (conf={CONF_THRESH})...\n")

    for idx, (images, _) in enumerate(test_loader):
        img_tensor = images.to(device)
        txt_path = Path("yolo_dataset_augmented/labels/test") / f"{test_dataset.img_paths[idx].stem}.txt"

        with torch.no_grad():
            logits, bbox_reg, centerness = model(img_tensor)

        preds_raw = decode_predictions(
            logits, bbox_reg, centerness, locations, conf_thresh=CONF_THRESH, nms_thresh=0.45
        )[0]

        # Собираем предсказания с координатами для фильтра
        preds_with_coords = []
        if len(preds_raw['boxes']) > 0:
            for i in range(len(preds_raw['boxes'])):
                box = preds_raw['boxes'][i].cpu().numpy()
                char = INV_CLASSES[int(preds_raw['labels'][i].cpu().item())]
                conf = preds_raw['scores'][i].cpu().item()
                cx = (box[0] + box[2]) / 2
                preds_with_coords.append((cx, char, box, conf))

        pred_chars = filter_plate_predictions(preds_with_coords)
        pred_str = "".join(pred_chars)
        gt_str = get_gt_from_txt(txt_path)

        is_correct = (pred_str == gt_str) and len(gt_str) > 0
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        logger.info(f"[{idx+1:2d}/{total}] {status} GT: {gt_str: <10} | Pred: {pred_str: <10}")

    accuracy = correct / total if total > 0 else 0
    logger.info("\n" + "=" * 55)
    logger.info(f"🎯 FCOS EXACT MATCH ACCURACY: {accuracy:.2%} ({correct}/{total})")
    logger.info("=" * 55)


if __name__ == "__main__":
    evaluate()
