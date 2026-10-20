"""FCOS (Fully Convolutional One-Stage) детектор для символов номеров."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d

# Strides уровней пирамиды P3, P4, P5
STRIDES = [8, 16, 32]


def compute_locations(features, strides=STRIDES):
    """Координаты центров ячеек каждого уровня пирамиды в пикселях входного изображения."""
    locations = []
    for feature, stride in zip(features, strides, strict=True):
        h, w = feature.shape[-2:]
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32, device=feature.device) + stride // 2
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32, device=feature.device) + stride // 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        locations.append(torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1))
    return locations


class FPN(nn.Module):
    """Feature Pyramid Network — создаёт пирамиду признаков разного масштаба."""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            layer_block = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            # Инициализация весов
            nn.init.kaiming_uniform_(inner_block.weight, a=1)
            nn.init.constant_(inner_block.bias, 0)
            nn.init.kaiming_uniform_(layer_block.weight, a=1)
            nn.init.constant_(layer_block.bias, 0)

            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, x):
        """
        x: список feature maps из backbone [C3, C4, C5]
        возвращает: список feature maps пирамиды [P3, P4, P5]
        """
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]

        # Идём снизу вверх, добавляем upsample + element-wise sum
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode='nearest')
            last_inner = inner_lateral + top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return results


class FCOSHead(nn.Module):
    """Голова детектора — предсказывает классы, боксы и centerness."""

    def __init__(self, in_channels=256, num_classes=27, num_convs=4):
        super().__init__()
        self.num_classes = num_classes

        # Классификационная ветвь
        cls_tower = []
        for _ in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Регрессионная ветвь (координаты бокса)
        bbox_tower = []
        for _ in range(num_convs):
            bbox_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU(inplace=True))
        self.bbox_tower = nn.Sequential(*bbox_tower)

        # Финальные слои предсказания
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)  # l, t, r, b
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # Инициализация весов как в FCOS/RetinaNet: маленькие начальные выходы стабилизируют exp в регрессии
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred, self.centerness]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.constant_(layer.bias, 0)

        # Инициализация bias для классификации (focal loss prior)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        """
        x: список feature maps из FPN
        возвращает: списки предсказаний для каждого уровня пирамиды
        """
        logits = []
        bbox_reg = []
        centerness = []

        for feature in x:
            cls_tower_out = self.cls_tower(feature)
            bbox_tower_out = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower_out))
            bbox_reg.append(self.bbox_pred(bbox_tower_out))
            centerness.append(self.centerness(bbox_tower_out))

        return logits, bbox_reg, centerness


class FCOSDetector(nn.Module):
    """Полная модель FCOS детектора."""

    def __init__(self, num_classes=27, backbone_pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: ResNet-18 с предобученными весами; BN заморожен - при batch=4 его статистика слишком шумная
        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT if backbone_pretrained else None, norm_layer=FrozenBatchNorm2d
        )

        # Извлекаем нужные слои (C3, C4, C5)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2  # C3: 128 channels
        self.layer3 = backbone.layer3  # C4: 256 channels
        self.layer4 = backbone.layer4  # C5: 512 channels

        # FPN
        self.fpn = FPN(in_channels_list=[128, 256, 512], out_channels=256)

        # Head
        self.head = FCOSHead(in_channels=256, num_classes=num_classes)

        # Learnable scale для регрессии на каждом уровне пирамиды
        self.scales = nn.ModuleList([Scale(init=1.0) for _ in STRIDES])

    def forward(self, x):
        """
        x: входное изображение (B, 3, H, W)
        возвращает: предсказания для каждого уровня пирамиды
        """
        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # FPN
        features = self.fpn([c3, c4, c5])

        # Head
        logits, bbox_reg, centerness = self.head(features)

        # Расстояния l, t, r, b в пикселях: exp(scale * x) * stride, как в FCOS
        bbox_reg = [
            torch.exp(scale(reg)) * stride for scale, reg, stride in zip(self.scales, bbox_reg, STRIDES, strict=True)
        ]

        return logits, bbox_reg, centerness


class Scale(nn.Module):
    """Learnable scale factor для bbox regression."""

    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale
