"""Постобработка предсказаний детектора символов номера."""

from statistics import median

# Символы номера одной высоты и на одной линии: допуски относительно медианной высоты
HEIGHT_TOLERANCE = 0.3
LINE_TOLERANCE = 0.4
MAX_PLATE_LENGTH = 7


def fit_line(xs, ys):
    """Прямая y = slope * x + intercept по оценке Тейла-Сена: отдельные ложные детекции её не сдвигают."""
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i]) for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] != xs[i]
    ]
    slope = median(slopes) if slopes else 0.0
    return slope, median(y - slope * x for x, y in zip(xs, ys, strict=True))


def sort_left_to_right(preds):
    """class_id всех предсказаний слева направо - без фильтра, для сравнения с ним."""
    return [class_id for _, class_id, _ in sorted(preds, key=lambda pred: pred[0][0] + pred[0][2])]


def filter_plate_predictions(preds):
    """Отбрасывает выбросы по высоте и линии, оставляет до 7 самых уверенных символов.

    preds: список (box [x1, y1, x2, y2], class_id, conf); возвращает class_id слева направо.
    """
    if not preds:
        return []

    heights = [box[3] - box[1] for box, _, _ in preds]
    centers_x = [(box[0] + box[2]) / 2 for box, _, _ in preds]
    centers_y = [(box[1] + box[3]) / 2 for box, _, _ in preds]
    med_h = median(heights)
    # Номер на кропе часто наклонён: линия символов - прямая через центры, а не горизонталь
    slope, intercept = fit_line(centers_x, centers_y)

    kept = [
        pred
        for pred, h, x, y in zip(preds, heights, centers_x, centers_y, strict=True)
        if abs(h - med_h) < HEIGHT_TOLERANCE * med_h and abs(y - (slope * x + intercept)) < LINE_TOLERANCE * med_h
    ]
    kept = sorted(kept, key=lambda pred: pred[2], reverse=True)[:MAX_PLATE_LENGTH]
    return sort_left_to_right(kept)
