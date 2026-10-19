"""Честное разделение train выборки на train (80%) и val (20%)."""

import random
import shutil
from pathlib import Path

from src.data.images import list_images

random.seed(42)

train_dir = Path("number_plates/train")
val_dir = Path("number_plates/val")
val_dir.mkdir(exist_ok=True)

# Находим все изображения
images = list_images(train_dir)
random.shuffle(images)

# Делим 80/20
val_size = int(0.2 * len(images))
val_images = images[:val_size]
train_images = images[val_size:]

print(f"Всего изображений: {len(images)}")
print(f"Перемещаем в val: {len(val_images)}")
print(f"Оставляем в train: {len(train_images)}")

# Перемещаем изображения вместе с аннотациями
for img_path in val_images:
    for src in [img_path, img_path.with_suffix(".txt"), img_path.with_suffix(".xml")]:
        shutil.move(str(src), str(val_dir / src.name))

print(f"✅ Разделение завершено! Перемещено изображений: {len(val_images)}")
