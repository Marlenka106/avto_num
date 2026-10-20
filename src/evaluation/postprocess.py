"""Постобработка предсказаний детектора символов номера."""

from statistics import median

# Символы номера одной высоты и на одной линии: допуски относительно медианной высоты
HEIGHT_TOLERANCE = 0.3
LINE_TOLERANCE = 0.4
MAX_PLATE_LENGTH = 7


def filter_plate_predictions(preds):
    """Отбрасывает выбросы по высоте и линии, оставляет до 7 самых уверенных символов.

    preds: список (box [x1, y1, x2, y2], class_id, conf); возвращает class_id слева направо.
    """
    if not preds:
        return []

    heights = [box[3] - box[1] for box, _, _ in preds]
    centers = [(box[1] + box[3]) / 2 for box, _, _ in preds]
    med_h, med_y = median(heights), median(centers)

    kept = [
        pred
        for pred, h, y in zip(preds, heights, centers, strict=True)
        if abs(h - med_h) < HEIGHT_TOLERANCE * med_h and abs(y - med_y) < LINE_TOLERANCE * med_h
    ]
    kept = sorted(kept, key=lambda pred: pred[2], reverse=True)[:MAX_PLATE_LENGTH]
    return [class_id for _, class_id, _ in sorted(kept, key=lambda pred: pred[0][0] + pred[0][2])]
