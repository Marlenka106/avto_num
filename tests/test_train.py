"""Тесты цикла валидации FCOS."""

import math

import torch

from src.models.fcos import FCOSDetector
from src.models.loss import FCOSLoss
from src.training.train import validate


def test_validate_returns_loss_and_exact_match():
    torch.manual_seed(0)
    model = FCOSDetector(num_classes=27, backbone_pretrained=False)
    images = torch.randn(2, 3, 64, 128)
    targets = [
        {'boxes': torch.tensor([[40.0, 8.0, 50.0, 56.0], [10.0, 8.0, 20.0, 56.0]]), 'labels': torch.tensor([3, 1])},
        {'boxes': torch.tensor([[60.0, 8.0, 70.0, 56.0]]), 'labels': torch.tensor([12])},
    ]

    val_loss, val_acc = validate(model, FCOSLoss(num_classes=27), [(images, targets)], torch.device('cpu'))

    assert math.isfinite(val_loss)
    assert val_acc == 0.0
