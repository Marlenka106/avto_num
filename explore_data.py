"""Разведочный анализ датасета: статистика в консоль и графики в reports/figures."""

import random
from collections import Counter
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from src.data.classes import CLASSES, read_plate
from src.data.images import list_images
from src.data.quality import is_clean_annotation

DATA_DIR = Path("number_plates")
FIGURES_DIR = Path("reports/figures")
SPLITS = ["train", "test"]
SEED = 42


def collect(split):
    """Строка номера, размер изображения и class_id символов по каждому изображению сплита."""
    rows = []
    for img_path in list_images(DATA_DIR / split):
        txt_path = img_path.with_suffix(".txt")
        class_ids = [int(line.split()[0]) for line in txt_path.read_text().splitlines()]
        rows.append(
            {
                "img_path": img_path,
                "plate": read_plate(txt_path),
                "size": Image.open(img_path).size,
                "class_ids": class_ids,
                "clean": is_clean_annotation(txt_path),
            }
        )
    return rows


def print_summary(data):
    """Ключевые числа, на которых строятся выводы EDA."""
    for split, rows in data.items():
        extensions = Counter(row["img_path"].suffix for row in rows)
        objects = sum(len(row["class_ids"]) for row in rows)
        clean = sum(row["clean"] for row in rows)
        print(f"{split}: {len(rows)} изображений {dict(extensions)}, {objects} символов, полная разметка {clean}")

    test_classes = {class_id for row in data["test"] for class_id in row["class_ids"]}
    print(f"Классы без примеров в test: {[CLASSES[i] for i in range(len(CLASSES)) if i not in test_classes]}")

    train_plates = Counter(row["plate"] for row in data["train"])
    print(f"Уникальных номеров в train: {len(train_plates)}, самые частые: {train_plates.most_common(3)}")
    seen = [row["plate"] for row in data["test"] if row["plate"] in train_plates]
    print(f"Номеров test, встречающихся в train: {len(seen)}/{len(data['test'])}")


def plot_class_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.4
    for offset, split in zip([-width / 2, width / 2], SPLITS, strict=True):
        counts = Counter(class_id for row in data[split] for class_id in row["class_ids"])
        ax.bar([i + offset for i in range(len(CLASSES))], [counts[i] for i in range(len(CLASSES))], width, label=split)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yscale("log")
    ax.set_title("Число символов каждого класса")
    ax.legend()
    return fig


def plot_label_length(data):
    fig, ax = plt.subplots(figsize=(8, 4))
    for split in SPLITS:
        lengths = Counter(len(row["class_ids"]) for row in data[split])
        total = len(data[split])
        ax.bar(
            [length + (0.2 if split == "test" else -0.2) for length in lengths],
            [n / total for n in lengths.values()],
            0.4,
            label=split,
        )
    ax.set_xlabel("Символов в разметке")
    ax.set_ylabel("Доля изображений")
    ax.set_title("Длина разметки: всё, кроме 7, - неполный номер или ошибка")
    ax.legend()
    return fig


def plot_image_sizes(data):
    fig, ax = plt.subplots(figsize=(7, 5))
    for split in SPLITS:
        widths, heights = zip(*(row["size"] for row in data[split]), strict=True)
        ax.scatter(widths, heights, s=8, alpha=0.6, label=split)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ширина, px")
    ax.set_ylabel("Высота, px")
    ax.set_title("Размеры изображений")
    ax.legend()
    return fig


def plot_plate_repetition(data):
    repeats = Counter(Counter(row["plate"] for row in data["train"]).values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(repeats), [count * n for count, n in repeats.items()])
    ax.set_xlabel("Сколько раз номер встречается в train")
    ax.set_ylabel("Изображений")
    ax.set_title("Повторы номеров: один автомобиль снят многократно")
    return fig


def plot_samples(data):
    """6 случайных изображений train с боксами и подписями символов."""
    rows = random.Random(SEED).sample(data["train"], 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    for ax, row in zip(axes.flatten(), rows, strict=True):
        img = Image.open(row["img_path"]).convert("RGB")
        w, h = img.size
        ax.imshow(img)
        for line in row["img_path"].with_suffix(".txt").read_text().splitlines():
            class_id, cx, cy, bw, bh = line.split()
            x1, y1 = (float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), float(bw) * w, float(bh) * h, linewidth=2, edgecolor="lime", facecolor="none"
                )
            )
            ax.text(
                x1,
                y1,
                CLASSES[int(class_id)],
                color="white",
                fontsize=10,
                fontweight="bold",
                bbox={"facecolor": "red", "alpha": 0.7, "pad": 1},
            )
        ax.set_title(row["plate"])
        ax.axis("off")
    return fig


def main():
    # 1. Сбор статистики
    data = {split: collect(split) for split in SPLITS}
    print_summary(data)

    # 2. Графики
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plots = {
        "class_distribution": plot_class_distribution,
        "label_length": plot_label_length,
        "image_sizes": plot_image_sizes,
        "plate_repetition": plot_plate_repetition,
        "sample_images": plot_samples,
    }
    for name, plot in plots.items():
        fig = plot(data)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=100)
        plt.close(fig)
        print(f"Сохранено: {FIGURES_DIR / name}.png")


if __name__ == "__main__":
    main()
