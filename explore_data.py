"""Разведочный анализ датасета: статистика в консоль и графики в reports/figures."""

import hashlib
import random
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from src.data.classes import CLASSES, LABEL_FIXES, read_label, read_plate
from src.data.images import list_images
from src.data.quality import is_clean_annotation

DATA_DIR = Path("number_plates")
FIGURES_DIR = Path("reports/figures")
SPLITS_DIR = Path("splits")
SPLITS = ["train", "test"]
SEED = 42
# SHA-256 данных, на которых получены результаты README и REPORT
DATASET_SHA256 = "db62504275571867fb5432c8a712af31a28a7686bc844a9a89e874401d8961c1"
# Допуск сверки рамок .xml и .txt: при конвертации VOC (с 1) в YOLO (с 0) центры сдвинуты на 1 px
BOX_TOLERANCE_PX = 2


def dataset_checksum():
    """SHA-256 всех файлов датасета: по нему проверяется, что данные те же самые."""
    digest = hashlib.sha256()
    for path in sorted(p for p in DATA_DIR.rglob("*") if p.is_file()):
        digest.update(path.relative_to(DATA_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_annotations(img_path):
    """Расхождения между .xml, .txt и изображением; пустое множество - разметка согласована."""
    width, height = Image.open(img_path).size
    root = ElementTree.parse(img_path.with_suffix(".xml")).getroot()
    xml_boxes = [
        (obj.findtext("name"), *(float(obj.findtext(f"bndbox/{key}")) for key in ["xmin", "ymin", "xmax", "ymax"]))
        for obj in root.iter("object")
    ]
    txt_boxes = []
    for line in img_path.with_suffix(".txt").read_text().splitlines():
        class_id, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        txt_boxes.append((CLASSES[int(class_id)], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))

    issues = set()
    if (int(root.findtext("size/width")), int(root.findtext("size/height"))) != (width, height):
        issues.add("размер в .xml не совпадает с изображением")
    if len(xml_boxes) != len(txt_boxes):
        issues.add("разное число рамок в .xml и .txt")
        return issues

    # Сравниваем рамки попарно слева направо
    for xml_box, txt_box in zip(
        sorted(xml_boxes, key=lambda b: b[1] + b[3]), sorted(txt_boxes, key=lambda b: b[1] + b[3]), strict=True
    ):
        if xml_box[0] != txt_box[0]:
            issues.add("символ в .xml не совпадает с классом в .txt")
        if max(abs(a - b) for a, b in zip(xml_box[1:], txt_box[1:], strict=True)) > BOX_TOLERANCE_PX:
            issues.add("рамки в .xml и .txt не совпадают")
        x1, y1, x2, y2 = txt_box[1:]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            issues.add("рамка выходит за край изображения")
    return issues


def collect(split):
    """Строка номера, размер изображения и class_id символов по каждому изображению сплита."""
    rows = []
    for img_path in list_images(DATA_DIR / split):
        txt_path = img_path.with_suffix(".txt")
        class_ids = [int(line.split()[0]) for line in read_label(txt_path)]
        rows.append(
            {
                "img_path": img_path,
                "plate": read_plate(txt_path),
                "size": Image.open(img_path).size,
                "class_ids": class_ids,
                "clean": is_clean_annotation(txt_path),
                "issues": check_annotations(img_path),
            }
        )
    return rows


def print_summary(data):
    """Ключевые числа, на которых строятся выводы EDA."""
    checksum = dataset_checksum()
    print(f"SHA-256 датасета: {checksum}")
    if checksum != DATASET_SHA256:
        print(f"⚠️  Данные отличаются от эталонных ({DATASET_SHA256}): результаты README и REPORT не повторятся")
    for split, rows in data.items():
        extensions = Counter(row["img_path"].suffix for row in rows)
        objects = sum(len(row["class_ids"]) for row in rows)
        clean = sum(row["clean"] for row in rows)
        print(f"{split}: {len(rows)} изображений {dict(extensions)}, {objects} символов, полная разметка {clean}")

        issues = Counter(issue for row in rows for issue in row["issues"])
        print(f"  сверка .xml/.txt/изображения: {dict(issues) or 'расхождений нет'}")

    print(f"Исправлено ошибок разметки: {len(LABEL_FIXES)} {list(LABEL_FIXES.values())}")
    coverage = class_coverage(data)
    rare = [c for i, c in enumerate(CLASSES) if min(coverage["val"][i], coverage["test"][i]) < 5]
    print("Символов по классам (train/val/test):")
    print(" ".join(f"{c}:{'/'.join(str(coverage[s][i]) for s in coverage)}" for i, c in enumerate(CLASSES)))
    print(f"Меньше 5 примеров в val или test: {rare}")

    test_classes = {class_id for row in data["test"] for class_id in row["class_ids"]}
    print(f"Классы без примеров в test: {[CLASSES[i] for i in range(len(CLASSES)) if i not in test_classes]}")

    train_plates = Counter(row["plate"] for row in data["train"])
    print(f"Уникальных номеров в train: {len(train_plates)}, самые частые: {train_plates.most_common(3)}")
    seen = [row["plate"] for row in data["test"] if row["plate"] in train_plates]
    print(f"Номеров test, встречающихся в train: {len(seen)}/{len(data['test'])}")


def class_coverage(data):
    """Число символов каждого класса в выборках, на которых модель учится и оценивается."""
    rows_by_name = {row["img_path"].name: row for row in data["train"]}
    splits = {
        name: [rows_by_name[n] for n in (SPLITS_DIR / f"{name}.txt").read_text().split()] for name in ["train", "val"]
    }
    splits["test"] = [row for row in data["test"] if row["clean"]]
    return {name: Counter(class_id for row in rows for class_id in row["class_ids"]) for name, rows in splits.items()}


def plot_class_distribution(data):
    coverage = class_coverage(data)
    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.27
    for offset, (split, counts) in zip([-width, 0, width], coverage.items(), strict=True):
        ax.bar([i + offset for i in range(len(CLASSES))], [counts[i] for i in range(len(CLASSES))], width, label=split)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yscale("log")
    ax.set_title("Число символов каждого класса (train/val по splits/, test с полной разметкой)")
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
        for line in read_label(row["img_path"].with_suffix(".txt")):
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
