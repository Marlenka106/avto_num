"""Тестирование кастомного Dataset и DataLoader (исправленная визуализация)."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CLASSES, PlateCharDataset, get_transform

# Пути к данным
train_img_dir = "number_plates/train"
train_txt_dir = "number_plates/train"

# 1. Создаем датасет БЕЗ аугментаций для честной проверки координат
print("Создание датасета (без аугментаций)...")
dataset = PlateCharDataset(
    img_dir=train_img_dir,
    txt_dir=train_txt_dir,
    transforms=get_transform(train=False),  # <-- ВАЖНО: train=False
)
print(f"✅ Датасет создан. Всего изображений: {len(dataset)}")

# 2. Создаем DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x, strict=False)))

# 3. Берем один батч
images, targets = next(iter(dataloader))
print(f"✅ Батч получен. Размер батча: {len(images)}")

# 4. Визуализируем ПЕРВЫЙ элемент батча
img_idx = 0
img_tensor = images[img_idx]
target = targets[img_idx]

# Получаем оригинальный путь к картинке, чтобы узнать её реальный размер
original_img_path = dataset.img_paths[dataset.indices[img_idx] if hasattr(dataset, 'indices') else img_idx]
# Проще: берем из самого датасета по индексу, но у нас shuffle=True, поэтому просто откроем по имени файла
# На самом деле, давайте просто выведем размер тензора
print(f"Форма тензора изображения: {img_tensor.shape} (C, H, W)")

# Денормализуем изображение для отображения (возвращаем к диапазону 0-1)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_display = img_tensor * std + mean
img_display = img_display.permute(1, 2, 0).clamp(0, 1).numpy()

fig, ax = plt.subplots(1, figsize=(10, 6))
ax.imshow(img_display)

print(f"Отрисовка {len(target['boxes'])} боксов...")
# Рисуем боксы из тензора
for i, box in enumerate(target["boxes"]):
    xmin, ymin, xmax, ymax = box.tolist()
    label_id = target["labels"][i].item()
    char = CLASSES[label_id]

    # Рисуем прямоугольник
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='cyan', facecolor='none')
    ax.add_patch(rect)

    # Подпись
    ax.text(
        xmin,
        ymin - 5,
        char,
        color='white',
        fontsize=12,
        fontweight='bold',
        bbox={"facecolor": 'magenta', "alpha": 0.7, "edgecolor": 'none', "pad": 2},
    )

ax.set_title("Проверка Dataset: координаты из .txt на исходном изображении")
ax.axis('off')

Path("reports/figures").mkdir(parents=True, exist_ok=True)
out_path = "reports/figures/dataset_check.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"✅ Визуализация сохранена в: {out_path}")
# plt.show() # Закомментировано, чтобы не было ворнинга в VS Code
