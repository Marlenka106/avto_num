"""Тесты декодирования предсказаний FCOS."""

import torch

from src.evaluation.inference import decode_predictions
from src.models.fcos import compute_locations


def test_decode_turns_confident_point_into_box():
    logits = torch.full((1, 27, 2, 4), -10.0)
    logits[0, 5, 0, 1] = 10.0
    bbox_reg = torch.full((1, 4, 2, 4), 10.0)
    centerness = torch.full((1, 1, 2, 4), 10.0)
    locations = compute_locations([logits], strides=[8])

    pred = decode_predictions([logits], [bbox_reg], [centerness], locations)[0]

    # Точка (12, 4) и расстояния l, t, r, b = 10
    assert pred['labels'].tolist() == [5]
    assert pred['boxes'].tolist() == [[2.0, -6.0, 22.0, 14.0]]
    assert pred['scores'].item() > 0.99


def test_decode_without_confident_points_returns_empty():
    logits = torch.full((1, 27, 2, 4), -10.0)
    reg = torch.ones(1, 4, 2, 4)

    pred = decode_predictions([logits], [reg], [torch.zeros(1, 1, 2, 4)], compute_locations([logits], strides=[8]))[0]

    assert len(pred['boxes']) == 0
