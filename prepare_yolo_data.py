"""Подготовка данных в формате YOLO."""

import shutil
from pathlib import Path

from src.data.build import staged_dataset
from src.data.classes import read_label
from src.data.images import list_images


def read_split(split_file, src_dir):
    """Читает список файлов сплита, созданный split_data.py."""
    return [src_dir / name for name in split_file.read_text().split()]


def prepare_yolo_dataset():
    dst_root = Path("yolo_dataset")
    with staged_dataset(dst_root) as staged:
        _prepare_yolo_dataset(staged)
    print(f"\n🎉 Датасет подготовлен в {dst_root}")


def _prepare_yolo_dataset(dst_root):
    # Исходные папки
    src_train = Path("number_plates/train")
    src_test = Path("number_plates/test")
    splits_dir = Path("splits")

    for split in ["train", "val", "test"]:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Функция копирования файлов
    def copy_split(img_paths, split_name):
        count = 0
        for img_path in img_paths:
            txt = img_path.with_suffix(".txt")

            # Копируем изображение
            shutil.copy(img_path, dst_root / "images" / split_name / img_path.name)
            # Копируем аннотацию с исправлением известных ошибок разметки
            (dst_root / "labels" / split_name / txt.name).write_text("".join(f"{line}\n" for line in read_label(txt)))
            count += 1

        print(f"✅ {split_name}: скопировано {count} пар (image + label)")

    # Копируем все сплиты
    copy_split(read_split(splits_dir / "train.txt", src_train), "train")
    copy_split(read_split(splits_dir / "val.txt", src_train), "val")
    copy_split(list_images(src_test), "test")


if __name__ == "__main__":
    prepare_yolo_dataset()
