"""Детерминированное разделение train выборки на train (80%) и val (20%) без утечки номеров."""

import random
from collections import defaultdict
from pathlib import Path

from src.data.images import list_images

TRAIN_DIR = Path("number_plates/train")
SPLITS_DIR = Path("splits")
VAL_FRACTION = 0.2
SEED = 42


def plate_key(txt_path):
    """Номер как последовательность class_id слева направо."""
    boxes = [line.split() for line in txt_path.read_text().splitlines()]
    return " ".join(box[0] for box in sorted(boxes, key=lambda box: float(box[1])))


def split_by_plate(images, val_fraction=VAL_FRACTION, seed=SEED):
    """Делит изображения так, чтобы один номер не попадал одновременно в train и val."""
    # Один автомобиль снят многократно: случайный сплит по изображениям даёт утечку номеров
    groups = defaultdict(list)
    for img_path in images:
        groups[plate_key(img_path.with_suffix(".txt"))].append(img_path)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    train, val = [], []
    for key in keys:
        target = val if len(val) < val_fraction * len(images) else train
        target.extend(groups[key])
    return sorted(train), sorted(val)


def main():
    images = list_images(TRAIN_DIR)
    train, val = split_by_plate(images)

    # Сохраняем списки файлов, исходные данные не трогаем
    SPLITS_DIR.mkdir(exist_ok=True)
    for name, split in [("train", train), ("val", val)]:
        (SPLITS_DIR / f"{name}.txt").write_text("".join(f"{p.name}\n" for p in split))

    print(f"Всего изображений: {len(images)}")
    print(f"train: {len(train)}, val: {len(val)}")
    print(f"✅ Списки сохранены в {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
