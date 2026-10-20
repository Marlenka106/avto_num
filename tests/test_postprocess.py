"""Тесты постобработки предсказаний."""

from src.evaluation.postprocess import filter_plate_predictions, fit_line, sort_left_to_right


def pred(x, class_id, conf=0.9, height=50, center_y=50):
    """Предсказание: бокс шириной 10 px с левым краем x."""
    return [x, center_y - height / 2, x + 10, center_y + height / 2], class_id, conf


def test_returns_class_ids_left_to_right():
    preds = [pred(30, 3), pred(10, 1), pred(20, 2)]

    assert filter_plate_predictions(preds) == [1, 2, 3]


def test_drops_boxes_with_outlier_height_or_line():
    preds = [pred(10 * i, i) for i in range(5)]
    preds += [pred(60, 20, height=100), pred(70, 21, height=20), pred(80, 22, center_y=100)]

    assert filter_plate_predictions(preds) == [0, 1, 2, 3, 4]


def test_keeps_seven_most_confident_characters():
    preds = [pred(10 * i, i, conf=0.5 + i / 100) for i in range(9)]

    assert filter_plate_predictions(preds) == [2, 3, 4, 5, 6, 7, 8]


def test_does_not_mutate_input():
    preds = [pred(30, 3), pred(10, 1)]
    snapshot = list(preds)

    filter_plate_predictions(preds)

    assert preds == snapshot


def test_no_detections():
    assert filter_plate_predictions([]) == []


def test_keeps_all_characters_of_tilted_plate():
    # Центры символов поднимаются на 15 px на каждый символ - больше допуска от горизонтали
    preds = [pred(20 * i, i, center_y=100 - 15 * i) for i in range(7)]

    assert filter_plate_predictions(preds) == list(range(7))


def test_drops_off_line_box_on_tilted_plate():
    preds = [pred(20 * i, i, center_y=100 - 15 * i) for i in range(6)] + [pred(70, 20, center_y=100)]

    assert filter_plate_predictions(preds) == list(range(6))


def test_fit_line_ignores_single_outlier():
    slope, intercept = fit_line([0, 10, 20, 30, 40], [5, 7, 9, 100, 13])

    assert (slope, intercept) == (0.2, 5.0)


def test_sort_left_to_right_keeps_all_predictions():
    preds = [pred(30, 3), pred(10, 1), pred(20, 2, height=200)]

    assert sort_left_to_right(preds) == [1, 2, 3]
