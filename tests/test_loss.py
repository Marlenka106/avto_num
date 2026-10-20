"""Тесты функции потерь FCOS."""

import pytest
import torch
from torchvision.ops import sigmoid_focal_loss

from src.models.loss import FCOSLoss, FocalLoss, IOULoss

NUM_CLASSES = 27


def test_focal_loss_matches_torchvision():
    torch.manual_seed(0)
    logits = torch.randn(16, NUM_CLASSES)
    targets = (torch.rand(16, NUM_CLASSES) > 0.9).float()

    expected = sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum")

    assert torch.allclose(FocalLoss()(logits, targets), expected, atol=1e-5)


class TestIOULoss:
    def test_identical_boxes_give_zero_loss(self):
        boxes = torch.tensor([[3.0, 4.0, 5.0, 6.0], [1.0, 1.0, 1.0, 1.0]])

        assert IOULoss()(boxes, boxes).item() == pytest.approx(0.0, abs=1e-5)

    def test_nested_boxes_give_one_minus_iou(self):
        # Боксы вокруг общей точки: площади 4 и 16, вложенные, поэтому GIoU = IoU = 0.25
        pred = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        target = torch.tensor([[2.0, 2.0, 2.0, 2.0]])

        assert IOULoss()(pred, target).item() == pytest.approx(0.75, abs=1e-5)

    def test_is_symmetric(self):
        a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        b = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

        assert IOULoss()(a, b).item() == pytest.approx(IOULoss()(b, a).item())


class TestComputeTargets:
    @pytest.fixture
    def loss_fn(self):
        return FCOSLoss(num_classes=NUM_CLASSES)

    def _targets(self, loss_fn, boxes, labels):
        locations = [torch.tensor([[float(x), float(y)] for y in range(0, 64, 8) for x in range(0, 64, 8)])]
        return loss_fn.compute_targets(locations, boxes[None], labels[None])

    def test_point_inside_box_is_positive_with_its_label(self, loss_fn):
        boxes = torch.tensor([[4.0, 4.0, 20.0, 20.0]])
        labels = torch.tensor([5])

        cls_t, box_t, _ = self._targets(loss_fn, boxes, labels)

        point_8_8 = 1 * 8 + 1
        assert cls_t[0, point_8_8].argmax().item() == 5
        assert cls_t[0, point_8_8].sum().item() == 1
        assert box_t[0, point_8_8].tolist() == [4.0, 4.0, 12.0, 12.0]

    def test_point_outside_all_boxes_is_negative(self, loss_fn):
        boxes = torch.tensor([[4.0, 4.0, 20.0, 20.0]])
        labels = torch.tensor([5])

        cls_t, _, ctr_t = self._targets(loss_fn, boxes, labels)

        point_40_40 = 5 * 8 + 5
        assert cls_t[0, point_40_40].sum().item() == 0
        assert ctr_t[0, point_40_40].item() == 0

    def test_smallest_box_wins_on_overlap(self, loss_fn):
        boxes = torch.tensor([[0.5, 0.5, 60.0, 60.0], [4.0, 4.0, 12.0, 12.0]])
        labels = torch.tensor([1, 2])

        cls_t, _, _ = self._targets(loss_fn, boxes, labels)

        point_8_8 = 1 * 8 + 1
        assert cls_t[0, point_8_8].argmax().item() == 2

    def test_centerness_is_one_at_box_center(self, loss_fn):
        boxes = torch.tensor([[0.5, 0.5, 15.5, 15.5]])
        labels = torch.tensor([0])

        _, _, ctr_t = self._targets(loss_fn, boxes, labels)

        point_8_8 = 1 * 8 + 1
        assert ctr_t[0, point_8_8].item() == pytest.approx(1.0, abs=1e-5)

    def test_object_is_assigned_to_pyramid_level_by_size(self, loss_fn):
        # Одна и та же точка на P3 и P4: max(l, t, r, b) = 98 попадает только в диапазон P4 (64-128)
        locations = [torch.tensor([[32.0, 32.0]]), torch.tensor([[32.0, 32.0]])]
        boxes = torch.tensor([[[0.0, 0.0, 130.0, 64.0]]])

        cls_t, _, _ = loss_fn.compute_targets(locations, boxes, torch.tensor([[7]]))

        assert cls_t[0, 0].sum().item() == 0
        assert cls_t[0, 1].argmax().item() == 7


def test_forward_returns_finite_losses_and_backpropagates():
    torch.manual_seed(0)
    shapes = [(8, 16), (4, 8), (2, 4)]
    logits = [torch.randn(2, NUM_CLASSES, h, w, requires_grad=True) for h, w in shapes]
    bbox_reg = [torch.rand(2, 4, h, w, requires_grad=True) for h, w in shapes]
    centerness = [torch.randn(2, 1, h, w, requires_grad=True) for h, w in shapes]
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 40.0, 60.0]]), "labels": torch.tensor([3])},
        {"boxes": torch.tensor([[60.0, 5.0, 90.0, 50.0]]), "labels": torch.tensor([20])},
    ]

    losses = FCOSLoss(num_classes=NUM_CLASSES)(logits, bbox_reg, centerness, targets)

    assert set(losses) == {"total_loss", "loss_cls", "loss_bbox", "loss_centerness"}
    assert all(torch.isfinite(v) for v in losses.values())
    losses["total_loss"].backward()
    assert all(t.grad is not None for t in logits)
