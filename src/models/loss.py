"""Полная функция потерь для FCOS детектора с matching algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fcos import compute_locations

# Уровень пирамиды отвечает за объекты с max(l, t, r, b) в своём диапазоне (P3, P4, P5), как в FCOS
SIZE_RANGES = [(0, 64), (64, 128), (128, float('inf'))]


class FocalLoss(nn.Module):
    """Focal Loss для классификации."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = focal_weight * ((1 - p_t) ** self.gamma)
        return (focal_weight * ce_loss).sum()


class IOULoss(nn.Module):
    """Корректный GIoU Loss для FCOS (расстояния от центра: left, top, right, bottom)."""

    def forward(self, pred, target):
        # pred, target: (N, 4) -> [l, t, r, b]
        # Конвертируем в координаты боксов относительно точки (0, 0)
        pred_x1, pred_y1 = -pred[:, 0], -pred[:, 1]
        pred_x2, pred_y2 = pred[:, 2], pred[:, 3]

        target_x1, target_y1 = -target[:, 0], -target[:, 1]
        target_x2, target_y2 = target[:, 2], target[:, 3]

        # Пересечение
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Площади
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-6)

        # Ограничивающий бокс (Enclosing box)
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)

        enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
        enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
        enclose_area = enclose_w * enclose_h

        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
        return (1 - giou).mean()


class FCOSLoss(nn.Module):
    """Полная функция потерь для FCOS с matching algorithm."""

    def __init__(self, num_classes=27):
        super().__init__()
        self.num_classes = num_classes
        self.focal_loss = FocalLoss()
        self.iou_loss = IOULoss()

    def compute_targets(self, locations, gt_boxes, gt_labels):
        """
        Matching algorithm: определяет, какие точки отвечают за каждый GT бокс.

        locations: список координат точек для каждого уровня [(N0, 2), (N1, 2), ...]
        gt_boxes: (B, M, 4) — [xmin, ymin, xmax, ymax]
        gt_labels: (B, M) — классы
        """
        B, M = gt_boxes.shape[:2]

        # Объединяем все точки со всех уровней
        all_locations = torch.cat(locations, dim=0)  # (N, 2)
        N = all_locations.shape[0]

        # Диапазон размеров объектов, за которые отвечает уровень каждой точки
        size_ranges = torch.cat(
            [loc.new_tensor(SIZE_RANGES[level]).expand(len(loc), 2) for level, loc in enumerate(locations)]
        )  # (N, 2)

        # Для каждого изображения вычисляем targets
        cls_targets = []
        box_targets = []
        centerness_targets = []

        for b in range(B):
            boxes = gt_boxes[b]  # (M, 4)
            labels = gt_labels[b]  # (M,)

            # Вычисляем расстояния от каждой точки до каждой GT коробки
            # locations: (N, 2), boxes: (M, 4)
            # Результат: (N, M, 4) — [left, top, right, bottom]
            xs, ys = all_locations[:, 0], all_locations[:, 1]

            left = xs[:, None] - boxes[None, :, 0]  # (N, M)
            top = ys[:, None] - boxes[None, :, 1]
            right = boxes[None, :, 2] - xs[:, None]
            bottom = boxes[None, :, 3] - ys[:, None]

            distances = torch.stack([left, top, right, bottom], dim=2)  # (N, M, 4)

            # Точка положительная, если все расстояния > 0 (т.е. точка внутри бокса)
            is_in_boxes = distances.min(dim=2)[0] > 0  # (N, M)

            # и объект соответствует уровню пирамиды по max(l, t, r, b)
            max_distance = distances.max(dim=2)[0]  # (N, M)
            is_cared_in_level = (max_distance >= size_ranges[:, None, 0]) & (max_distance <= size_ranges[:, None, 1])

            # Для каждой точки выбираем GT бокс с минимальной площадью (если точка в нескольких боксах)
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (M,)
            box_areas = box_areas[None].expand(N, M)  # (N, M)

            # Маскируем боксы, в которые точка не попадает или которые не относятся к её уровню
            inf_tensor = torch.full_like(box_areas, float('inf'))
            box_areas = torch.where(is_in_boxes & is_cared_in_level, box_areas, inf_tensor)

            # Для каждой точки выбираем бокс с минимальной площадью
            min_area, min_area_idx = box_areas.min(dim=1)  # (N,)

            # Определяем, является ли точка положительной
            is_positive = min_area < float('inf')  # (N,)

            # Формируем targets
            cls_target = torch.zeros(N, self.num_classes, device=boxes.device)
            box_target = torch.zeros(N, 4, device=boxes.device)
            centerness_target = torch.zeros(N, device=boxes.device)

            if is_positive.any():
                pos_idx = torch.where(is_positive)[0]
                matched_boxes = min_area_idx[pos_idx]

                # Классы (one-hot encoding)
                matched_labels = labels[matched_boxes]
                cls_target[pos_idx, matched_labels] = 1.0

                # Боксы (расстояния до границ)
                box_target[pos_idx] = distances[pos_idx, matched_boxes]

                # Centerness: sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
                l, t, r, b = box_target[pos_idx].unbind(dim=1)
                centerness_target[pos_idx] = torch.sqrt(
                    (torch.min(l, r) / (torch.max(l, r) + 1e-6)) * (torch.min(t, b) / (torch.max(t, b) + 1e-6))
                )

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            centerness_targets.append(centerness_target)

        return torch.stack(cls_targets), torch.stack(box_targets), torch.stack(centerness_targets)

    def forward(self, logits, bbox_reg, centerness, targets):
        """
        logits: список предсказаний классов [(B, C, H, W), ...]
        bbox_reg: список предсказаний боксов [(B, 4, H, W), ...]
        centerness: список предсказаний centerness [(B, 1, H, W), ...]
        targets: список таргетов для каждого изображения
                 [{'boxes': (M, 4), 'labels': (M,)}, ...]
        """
        # Генерируем locations
        locations = compute_locations(logits)

        # Подготавливаем GT боксы и labels
        B = len(targets)
        max_boxes = max(len(t['boxes']) for t in targets)
        if max_boxes == 0:
            max_boxes = 1  # защита от пустых таргетов

        gt_boxes = torch.zeros(B, max_boxes, 4, device=logits[0].device)
        gt_labels = torch.zeros(B, max_boxes, dtype=torch.long, device=logits[0].device)

        for b, target in enumerate(targets):
            n_boxes = len(target['boxes'])
            if n_boxes > 0:
                gt_boxes[b, :n_boxes] = target['boxes']
                gt_labels[b, :n_boxes] = target['labels']

        # Вычисляем targets для всех точек
        cls_targets, box_targets, centerness_targets = self.compute_targets(locations, gt_boxes, gt_labels)

        # Форматируем предсказания: склеиваем все уровни пирамиды
        # Классы: (B, N, C)
        logits_flat = torch.cat([log.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes) for log in logits], dim=1)

        # Боксы: (B, N, 4)
        bbox_flat = torch.cat([reg.permute(0, 2, 3, 1).reshape(B, -1, 4) for reg in bbox_reg], dim=1)

        # Centerness: (B, N)
        centerness_flat = torch.cat([cent.permute(0, 2, 3, 1).reshape(B, -1) for cent in centerness], dim=1)

        # Вычисляем loss
        # Classification loss
        num_pos = max(1, cls_targets.sum().item())
        loss_cls = self.focal_loss(logits_flat, cls_targets) / num_pos

        # Regression loss (только для положительных точек)
        pos_mask = cls_targets.sum(dim=2) > 0  # (B, N)
        if pos_mask.any():
            loss_bbox = self.iou_loss(bbox_flat[pos_mask], box_targets[pos_mask])
            loss_centerness = F.binary_cross_entropy_with_logits(
                centerness_flat[pos_mask], centerness_targets[pos_mask], reduction='mean'
            )
        else:
            loss_bbox = bbox_flat.sum() * 0
            loss_centerness = centerness_flat.sum() * 0

        total_loss = loss_cls + loss_bbox + loss_centerness

        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_centerness': loss_centerness,
        }
