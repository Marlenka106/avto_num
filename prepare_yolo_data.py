"""Подготовка данных в формате YOLO."""

import shutil
from pathlib import Path

from src.data.classes import read_label
from src.data.images import list_images
from src.data.yolo_config import write_data_yaml


def read_split(split_file, src_dir):
    """Читает список файлов сплита, созданный split_data.py."""
    return [src_dir / name for name in split_file.read_text().split()]


def prepare_yolo_dataset():
    # Исходные папки
    src_train = Path("number_plates/train")
    src_test = Path("number_plates/test")
    splits_dir = Path("splits")

    # Целевая структура (пересоздаём, чтобы не осталось файлов от прошлого сплита)
    dst_root = Path("yolo_dataset")
    shutil.rmtree(dst_root, ignore_errors=True)

    for split in ["train", "val", "test"]:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Функция копирования файлов
    def copy_split(img_paths, split_name):
        count = 0
        for img_path in img_paths:
            txt = img_path.with_suffix(".txt")

            if not txt.exists():
                print(f"⚠️  Пропущен {img_path.stem}: нет .txt файла")
                continue

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
    write_data_yaml(dst_root)

    print(f"\n🎉 Датасет подготовлен в {dst_root}")


if __name__ == "__main__":
    prepare_yolo_dataset()
