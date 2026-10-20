"""PyTorch Dataset символов номерных знаков для обучения FCOS."""

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.data.images import list_images

INPUT_HEIGHT = 256
INPUT_WIDTH = 512


def get_transform(train):
    """Ресайз до 256x512 и нормализация ImageNet; для обучения - лёгкие аугментации."""
    augmentations = []
    if train:
        # Без отражений: зеркальный символ - это другой глиф
        augmentations = [
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.MotionBlur(blur_limit=5, p=0.3),
        ]

    return A.Compose(
        [A.Resize(INPUT_HEIGHT, INPUT_WIDTH), *augmentations, A.Normalize(), ToTensorV2()],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
    )


class PlateCharDataset(Dataset):
    """Кропы номеров с боксами символов из YOLO .txt аннотаций."""

    def __init__(self, img_dir, txt_dir, transforms):
        self.img_dir = Path(img_dir)
        self.txt_dir = Path(txt_dir)
        self.transforms = transforms
        self.img_paths = list_images(self.img_dir)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        height, width = image.shape[:2]
        boxes, labels = self._load_annotations(img_path.stem, width, height)

        transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
        target = {
            'boxes': torch.tensor(transformed['bboxes'], dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(transformed['labels'], dtype=torch.long),
        }
        return transformed['image'], target

    def _load_annotations(self, stem, width, height):
        """Читает YOLO-разметку и переводит её в пиксельные [xmin, ymin, xmax, ymax]."""
        boxes, labels = [], []
        for line in (self.txt_dir / f'{stem}.txt').read_text().splitlines():
            class_id, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
            # В разметке есть боксы, выходящие за край изображения на 1-2 px
            boxes.append(
                [
                    max(0.0, cx - w / 2),
                    max(0.0, cy - h / 2),
                    min(width, cx + w / 2),
                    min(height, cy + h / 2),
                ]
            )
            labels.append(int(class_id))
        return boxes, labels
