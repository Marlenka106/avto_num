"""Скрипт аугментации данных для увеличения датасета и борьбы с переобучением."""

import shutil
from pathlib import Path

import albumentations as A
import cv2

from src.data.images import list_images

# Пути
SRC_ROOT = Path("yolo_dataset")
OUT_ROOT = Path("yolo_dataset_augmented")
SRC_IMAGES = SRC_ROOT / "images" / "train"
SRC_LABELS = SRC_ROOT / "labels" / "train"
OUT_IMAGES = OUT_ROOT / "images" / "train"
OUT_LABELS = OUT_ROOT / "labels" / "train"


def get_transform():
    """Пайплайн аугментаций, оптимизированный для текста/номеров."""
    return A.Compose(
        [
            # 1. Геометрические искажения (имитация угла камеры)
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.7),
            # 2. Оптические искажения (имитация плохой фокусировки или движения)
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussianBlur(blur_limit=(3, 3), p=0.3),
            # 3. Изменение освещения и цвета (имитация времени суток, теней)
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # 4. Шум (имитация плохой камеры или сжатия JPEG)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        ],
        # В разметке есть боксы, выходящие за край изображения на 1-2 px
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True),
        seed=42,
    )


def augment_dataset():
    # Пересоздаём выходную папку, чтобы не осталось файлов от прошлого запуска
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    OUT_IMAGES.mkdir(parents=True)
    OUT_LABELS.mkdir(parents=True)

    # Копируем val и test без изменений (они должны оставаться "чистыми")
    for split in ["val", "test"]:
        for kind in ["images", "labels"]:
            shutil.copytree(SRC_ROOT / kind / split, OUT_ROOT / kind / split)

    # Новый пайплайн на каждый запуск, чтобы seed давал одинаковый результат
    transform = get_transform()
    image_files = list_images(SRC_IMAGES)
    print(f"Найдено {len(image_files)} исходных изображений для аугментации.")

    total_generated = 0

    for img_path in image_files:
        stem = img_path.stem
        label_path = SRC_LABELS / f"{stem}.txt"

        if not label_path.exists():
            continue

        # Читаем исходные боксы и классы
        bboxes = []
        class_labels = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w, h = map(float, parts)
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(cls))

        # Читаем изображение
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Копируем оригинал в новый датасет (без повторного JPEG-сжатия)
        shutil.copy(img_path, OUT_IMAGES / img_path.name)
        shutil.copy(label_path, OUT_LABELS / label_path.name)
        total_generated += 1

        # 2. Генерируем 3 аугментированные версии для каждого изображения
        for i in range(3):
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']

            if len(aug_bboxes) == 0:
                continue  # Если все боксы обрезались, пропускаем

            # Сохраняем новое изображение
            new_name = f"{stem}_aug_{i}.jpeg"
            cv2.imwrite(str(OUT_IMAGES / new_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            # Сохраняем новые лейблы
            new_label_name = f"{stem}_aug_{i}.txt"
            with open(OUT_LABELS / new_label_name, 'w') as f:
                for bbox, label in zip(aug_bboxes, aug_labels, strict=True):
                    f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

            total_generated += 1

    print(f"✅ Аугментация завершена! Всего изображений в train: {total_generated}")


if __name__ == "__main__":
    augment_dataset()
