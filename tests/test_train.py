"""Тесты цикла валидации FCOS."""

import math

import pytest
import torch
from PIL import Image
from torchvision.ops import box_iou

from evaluate import evaluate, make_fcos_predictor
from src.data.classes import CLASSES
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


class FixedFCOS(torch.nn.Module):
    """Две уверенные детекции с управляемым перекрытием на первом уровне FPN."""

    def __init__(self, box_width):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.box_width = box_width

    def forward(self, images):
        shapes = [(1, 2), (1, 1), (1, 1)]
        logits = [images.new_full((1, 27, h, w), -20.0) for h, w in shapes]
        bbox_reg = [images.new_ones((1, 4, h, w)) for h, w in shapes]
        centerness = [images.new_full((1, 1, h, w), 10.0) for h, w in shapes]
        logits[0][0, 1, 0, 0] = 10.0
        logits[0][0, 2, 0, 1] = 9.0
        # Центры ячеек (4, 4) и (12, 4): боксы сдвинуты по X на 8 px.
        bbox_reg[0][0] = images.new_tensor([4.0, 4.0, self.box_width - 4.0, 16.0]).view(4, 1, 1)
        return logits, bbox_reg, centerness


@pytest.mark.parametrize(('box_width', 'expected_labels'), [(20.0, [1, 2]), (24.0, [1])])
def test_validation_and_evaluation_use_same_nms(tmp_path, box_width, expected_labels):
    model = FixedFCOS(box_width)
    img_path = tmp_path / 'plate.png'
    Image.new('RGB', (128, 64), 'white').save(img_path)
    boxes = torch.tensor([[0.0, 0.0, box_width, 20.0], [8.0, 0.0, 8.0 + box_width, 20.0]])
    overlap = box_iou(boxes[:1], boxes[1:]).item()
    if len(expected_labels) == 2:
        assert 0.4 < overlap < 0.45
    else:
        assert overlap > 0.45

    target = {'boxes': boxes[: len(expected_labels)], 'labels': torch.tensor(expected_labels)}
    img_path.with_suffix('.txt').write_text(
        ''.join(f'{label} {0.1 * (i + 1)} 0.5 0.1 0.6\n' for i, label in enumerate(expected_labels))
    )
    predict = make_fcos_predictor(model.eval(), conf=0.3)

    # Проверяем реальные пути вызова NMS в оценке и выборе чекпоинта.
    assert [class_id for _, class_id, _ in predict(img_path)] == expected_labels
    rows = evaluate([img_path], tmp_path, predict, train_plates=set())
    val_loss, val_acc = validate(
        model, FCOSLoss(num_classes=27), [(torch.zeros(1, 3, 64, 128), [target])], torch.device('cpu')
    )

    assert rows[0]['pred'] == rows[0]['gt'] == ''.join(CLASSES[label] for label in expected_labels)
    assert math.isfinite(val_loss)
    assert val_acc == 1.0
