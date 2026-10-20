"""Тесты архитектуры FCOS."""

import pytest
import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from src.models.fcos import FPN, STRIDES, FCOSDetector, FCOSHead, Scale, compute_locations

NUM_CLASSES = 27


@pytest.fixture
def model():
    torch.manual_seed(0)
    return FCOSDetector(num_classes=NUM_CLASSES, backbone_pretrained=False)


def test_output_shapes_per_level(model):
    images = torch.randn(2, 3, 64, 128)

    logits, bbox_reg, centerness = model(images)

    assert len(logits) == len(bbox_reg) == len(centerness) == len(STRIDES)
    for stride, log, reg, cent in zip(STRIDES, logits, bbox_reg, centerness, strict=True):
        h, w = 64 // stride, 128 // stride
        assert log.shape == (2, NUM_CLASSES, h, w)
        assert reg.shape == (2, 4, h, w)
        assert cent.shape == (2, 1, h, w)


def test_bbox_regression_is_positive_and_finite(model):
    model.eval()
    with torch.no_grad():
        _, bbox_reg, _ = model(torch.randn(1, 3, 64, 128))

    for reg in bbox_reg:
        assert torch.isfinite(reg).all()
        assert (reg > 0).all()


def test_classification_bias_uses_focal_prior():
    head = FCOSHead(in_channels=256, num_classes=NUM_CLASSES)

    prior = torch.sigmoid(head.cls_logits.bias)

    assert torch.allclose(prior, torch.full_like(prior, 0.01), atol=1e-6)


def test_fpn_returns_same_resolution_as_inputs():
    fpn = FPN(in_channels_list=[8, 16], out_channels=32)
    features = [torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8)]

    outputs = fpn(features)

    assert [o.shape for o in outputs] == [(1, 32, 16, 16), (1, 32, 8, 8)]


def test_scale_multiplies_by_learnable_parameter():
    scale = Scale(init=2.5)

    assert torch.allclose(scale(torch.ones(3)), torch.full((3,), 2.5))
    assert scale.scale.requires_grad


def test_backward_produces_gradients(model):
    logits, bbox_reg, centerness = model(torch.randn(1, 3, 64, 128))
    loss = sum(t.mean() for t in [*logits, *bbox_reg, *centerness])

    loss.backward()

    assert all(p.grad is not None for p in model.head.parameters())


def test_backbone_batchnorm_is_frozen(model):
    modules = list(model.modules())

    assert not any(isinstance(m, nn.BatchNorm2d) for m in modules)
    assert any(isinstance(m, FrozenBatchNorm2d) for m in modules)


def test_locations_are_cell_centers():
    features = [torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 1, 2)]

    level0, level1 = compute_locations(features, strides=[8, 16])

    assert level0.tolist() == [[4, 4], [12, 4], [20, 4], [4, 12], [12, 12], [20, 12]]
    assert level1.tolist() == [[8, 8], [24, 8]]
