"""Первый взгляд на данные: визуализация изображений и bounding box."""

import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from src.data.images import list_images

# Пути к данным
train_dir = Path("number_plates/train")
test_dir = Path("number_plates/test")

# 1. Собираем статистику по классам из XML (чтобы знать имена символов)
all_chars = []
xml_files = list(train_dir.glob("*.xml"))
for xml_path in xml_files:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        all_chars.append(obj.find("name").text)

char_counts = Counter(all_chars)
print(f"Всего изображений в train: {len(list_images(train_dir))}")
print(f"Всего изображений в test:  {len(list_images(test_dir))}")
print(f"Уникальные символы ({len(char_counts)}): {sorted(char_counts.keys())}")
print(f"Топ-5 частых символов: {char_counts.most_common(5)}\n")

# 2. Визуализация 6 случайных примеров
sample_images = random.sample(list_images(train_dir), 6)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, img_path in enumerate(sample_images):
    # Загружаем изображение
    img = Image.open(img_path)

    # Загружаем аннотации из XML
    xml_path = img_path.with_suffix(".xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Рисуем
    ax = axes[idx]
    ax.imshow(img)

    # Рисуем bounding box для каждого символа
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        x1 = float(bbox.find("xmin").text)
        y1 = float(bbox.find("ymin").text)
        x2 = float(bbox.find("xmax").text)
        y2 = float(bbox.find("ymax").text)
        char = obj.find("name").text

        # Рисуем прямоугольник
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        # Подписываем символ
        ax.text(
            x1,
            y1 - 5,
            char,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox={"facecolor": 'red', "alpha": 0.7, "edgecolor": 'none', "pad": 1},
        )

    ax.set_title(img_path.stem[:15] + "...")
    ax.axis('off')

plt.tight_layout()

# Создаем папку для отчетов, если её нет
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# Сохраняем и показываем
out_path = "reports/figures/sample_images.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"✅ Визуализация сохранена в: {out_path}")
plt.show()
