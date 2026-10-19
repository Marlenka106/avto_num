"""Честное разделение train выборки на train (80%) и val (20%)."""

import random
import shutil
from pathlib import Path

random.seed(42)

train_dir = Path("number_plates/train")
val_dir = Path("number_plates/val")
val_dir.mkdir(exist_ok=True)

# Находим все уникальные имена файлов (без расширения)
stems = list({p.stem for p in train_dir.glob("*.jpeg")})
random.shuffle(stems)

# Делим 80/20
val_size = int(0.2 * len(stems))
val_stems = stems[:val_size]
train_stems = stems[val_size:]

print(f"Всего изображений: {len(stems)}")
print(f"Перемещаем в val: {len(val_stems)}")
print(f"Оставляем в train: {len(train_stems)}")

# Перемещаем файлы
moved_count = 0
for stem in val_stems:
    for ext in [".jpeg", ".txt", ".xml"]:
        src = train_dir / f"{stem}{ext}"
        dst = val_dir / f"{stem}{ext}"
        if src.exists():
            shutil.move(str(src), str(dst))
            moved_count += 1

print(f"✅ Разделение завершено! Перемещено файлов: {moved_count // 3}")
