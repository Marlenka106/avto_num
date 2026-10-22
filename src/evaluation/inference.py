"""Инференс и оценка точности распознавания номеров (Exact Match Accuracy)."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torchvision.ops as ops
from loguru import logger
from torch.utils.data import DataLoader

from src.data.classes import CLASSES, read_plate
from src.data.dataset import PlateCharDataset, get_transform
from src.data.quality import is_clean_annotation
from src.models.fcos import FCOSDetector, compute_locations
from src.utils.seed import set_seed

# Обратный маппинг: индекс класса -> символ
INV_CLASSES = dict(enumerate(CLASSES))
# Единый порог NMS для выбора чекпоинта, оценки и демонстрации FCOS.
NMS_THRESHOLD = 0.45


def collate_fn(batch):
    """Возвращает тензор изображений и список словарей-таргетов."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def filter_plate_chars(boxes, scores, labels):
    """Геометрическая фильтрация: символы номера равной высоты, на одной строке, без перекрытий."""
    if len(boxes) == 0:
        return boxes, scores, labels

    h = boxes[:, 3] - boxes[:, 1]
    cy = (boxes[:, 1] + boxes[:, 3]) / 2

    med_h = h.median()
    med_cy = cy.median()

    # 1) Размер: выбрасываем аномально мелкие/крупные
    m = (h > 0.5 * med_h) & (h < 2.0 * med_h)
    # 2) Одна строка: все символы на одном уровне
    m &= torch.abs(cy - med_cy) < 0.7 * med_h

    boxes, scores, labels = boxes[m], scores[m], labels[m]
    if len(boxes) < 2:
        return boxes, scores, labels

    # 3) Сортируем по левому краю и схлопываем перекрывающиеся по X
    order = boxes[:, 0].argsort()
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    keep = [0]
    for i in range(1, len(boxes)):
        j = keep[-1]
        inter = min(boxes[i, 2].item(), boxes[j, 2].item()) - max(boxes[i, 0].item(), boxes[j, 0].item())
        wi = (boxes[i, 2] - boxes[i, 0]).item()
        wj = (boxes[j, 2] - boxes[j, 0]).item()
        if inter > 0.35 * min(wi, wj):  # сильное перекрытие => это один и тот же символ
            if scores[i].item() > scores[j].item():
                keep[-1] = i  # оставляем более уверенный
            continue
        keep.append(i)

    idx = torch.tensor(keep, device=boxes.device)
    return boxes[idx], scores[idx], labels[idx]


def decode_predictions(
    logits, bbox_reg, centerness, locations, conf_thresh=0.3, nms_thresh=NMS_THRESHOLD, plate_filter=True
):
    """CLASS-AGNOSTIC декодирование FCOS + геометрическая фильтрация (plate_filter=False - только NMS)."""
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

            # Скор как в FCOS: среднее геометрическое уверенности класса и centerness
            scores_lvl = torch.sqrt(torch.sigmoid(log) * torch.sigmoid(cent).unsqueeze(1))
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

        # *** ГЛАВНОЕ ИСПРАВЛЕНИЕ: NMS для ВСЕХ классов вместе ***
        keep = ops.nms(boxes, scores, nms_thresh)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Геометрическая фильтрация по структуре номера
        if plate_filter:
            boxes, scores, labels = filter_plate_chars(boxes, scores, labels)

        order = boxes[:, 0].argsort()
        all_predictions.append(
            {
                'boxes': boxes[order],
                'scores': scores[order],
                'labels': labels[order],
            }
        )

    return all_predictions


def to_detections(pred):
    """Предсказания одного изображения как список (box [x1, y1, x2, y2], class_id, conf)."""
    return list(zip(pred['boxes'].tolist(), pred['labels'].tolist(), pred['scores'].tolist(), strict=True))


def evaluate():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. Загружаем тестовый датасет
    test_dataset = PlateCharDataset(
        img_dir="number_plates/test", txt_dir="number_plates/test", transforms=get_transform(train=False)
    )
    # Неполная разметка делает правильный ответ модели «ошибкой» - такие изображения не оцениваем
    test_dataset.img_paths = [
        p for p in test_dataset.img_paths if is_clean_annotation(test_dataset.txt_dir / f"{p.stem}.txt")
    ]
    # ДОБАВЛЕНО: collate_fn=collate_fn
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 2. Загружаем модель и веса
    model = FCOSDetector(num_classes=27, backbone_pretrained=False).to(device)
    weights_path = Path("weights/best.pt")
    if not weights_path.exists():
        logger.error(f"Weights not found at {weights_path}. Train the model first!")
        return

    # ДОБАВЛЕНО: weights_only=True для устранения FutureWarning
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Loaded weights from {weights_path}")

    # 3. Инференс
    correct_plates = 0
    total_plates = len(test_dataset)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    logger.info("Running inference on test set...")
    for idx, (images, _) in enumerate(test_loader):
        img_tensor = images.to(device)
        img_path = test_dataset.img_paths[idx]
        txt_path = test_dataset.txt_dir / f"{img_path.stem}.txt"

        with torch.no_grad():
            logits, bbox_reg, centerness = model(img_tensor)

        preds = decode_predictions(logits, bbox_reg, centerness, compute_locations(logits))[0]

        # Собираем предсказанную строку (сортируем по x1)
        pred_str = ""
        if len(preds['boxes']) > 0:
            order = preds['boxes'][:, 0].argsort()
            pred_labels = preds['labels'][order].cpu().numpy()
            pred_str = "".join([INV_CLASSES[int(lbl)] for lbl in pred_labels])

        # Ground Truth строка
        gt_str = read_plate(txt_path)

        if pred_str == gt_str and len(gt_str) > 0:
            correct_plates += 1
            status = "✅"
        else:
            status = "❌"

        logger.info(f"[{idx+1}/{total_plates}] {status} GT: {gt_str: <10} | Pred: {pred_str: <10}")

        # Визуализация первых 5 примеров
        if idx < 5:
            fig, ax = plt.subplots(1, figsize=(8, 4))
            img_display = img_tensor[0].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = (img_display * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(img_display)

            for i, box in enumerate(preds['boxes']):
                x1, y1, x2, y2 = box.cpu().numpy()
                lbl = INV_CLASSES[int(preds['labels'][i].cpu().item())]
                score = preds['scores'][i].cpu().item()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(
                    x1,
                    y1 - 2,
                    f"{lbl} {score:.2f}",
                    color='white',
                    fontsize=8,
                    bbox={'facecolor': 'red', 'alpha': 0.7, 'pad': 1},
                )

            ax.set_title(f"GT: {gt_str} | Pred: {pred_str}")
            ax.axis('off')
            plt.savefig(f"reports/figures/test_pred_{idx}.png", dpi=150, bbox_inches='tight')
            plt.close()

    accuracy = correct_plates / total_plates
    logger.success(f"🎯 EXACT MATCH ACCURACY: {accuracy:.2%} ({correct_plates}/{total_plates})")

    if accuracy >= 0.80:
        logger.success("🏆 ЦЕЛЬ ДОСТИГНУТА: Точность >= 80%!")
    else:
        logger.warning("⚠️ Точность < 80%. Рекомендуется дообучить модель (увеличить epochs до 50-100).")


if __name__ == "__main__":
    evaluate()
