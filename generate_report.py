"""Генерация всех визуализаций для финального отчета."""

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from ultralytics import YOLO

from evaluate_yolo_m import filter_plate_predictions, get_gt_from_txt
from src.data.dataset import CLASSES

# Настройки графиков
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

REPORT_DIR = Path("reports/final")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🎨 Генерация визуализаций для отчета")
print("=" * 60)

# ============================================================
# 1. РАСПРЕДЕЛЕНИЕ КЛАССОВ В ДАТАСЕТЕ
# ============================================================
print("\n📊 1/7: Распределение классов...")

all_chars = []
for xml_path in Path("number_plates/train").glob("*.xml"):
    tree = ET.parse(xml_path)
    for obj in tree.getroot().findall("object"):
        all_chars.append(obj.find("name").text)

char_counts = Counter(all_chars)
df_dist = pd.DataFrame({'Символ': list(char_counts.keys()), 'Количество': list(char_counts.values())}).sort_values(
    'Символ'
)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['#2ecc71' if c.isdigit() else '#3498db' for c in df_dist['Символ']]
ax.bar(df_dist['Символ'], df_dist['Количество'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Класс символа', fontsize=12)
ax.set_ylabel('Количество в train-наборе', fontsize=12)
ax.set_title('Распределение классов в исходном датасете (349 изображений)', fontsize=14, fontweight='bold')
ax.axhline(
    df_dist['Количество'].mean(), color='red', linestyle='--', label=f'Среднее: {df_dist["Количество"].mean():.1f}'
)
ax.legend()

# Легенда
legend_elements = [
    Patch(facecolor='#2ecc71', label='Цифры (10 классов)'),
    Patch(facecolor='#3498db', label='Буквы (17 классов)'),
]
ax.legend(handles=[*legend_elements, plt.Line2D([0], [0], color='red', linestyle='--', label='Среднее')])

plt.tight_layout()
plt.savefig(REPORT_DIR / "01_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Сохранено: {REPORT_DIR / '01_class_distribution.png'}")

# ============================================================
# 2. ГРАФИКИ ОБУЧЕНИЯ YOLOv8s (первая модель)
# ============================================================
print("\n📈 2/7: Кривые обучения YOLOv8s...")

results_path_1 = Path("runs/detect/runs/yolo/plates_detector/results.csv")
if results_path_1.exists():
    df_yolo_s = pd.read_csv(results_path_1)
    df_yolo_s.columns = df_yolo_s.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # mAP50
    axes[0].plot(df_yolo_s['epoch'], df_yolo_s['metrics/mAP50(B)'], 'o-', color='#27ae60', linewidth=2, markersize=4)
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('mAP@0.5')
    axes[0].set_title('YOLOv8s: mAP@0.5 (val)')
    axes[0].axhline(0.80, color='red', linestyle='--', alpha=0.5, label='Цель 80%')
    axes[0].legend()

    # Precision-Recall
    axes[1].plot(
        df_yolo_s['epoch'], df_yolo_s['metrics/precision(B)'], 'o-', label='Precision', color='#3498db', linewidth=2
    )
    axes[1].plot(df_yolo_s['epoch'], df_yolo_s['metrics/recall(B)'], 'o-', label='Recall', color='#e74c3c', linewidth=2)
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Значение')
    axes[1].set_title('Precision / Recall')
    axes[1].legend()

    # Losses
    axes[2].plot(df_yolo_s['epoch'], df_yolo_s['train/box_loss'], label='train box', color='#3498db')
    axes[2].plot(df_yolo_s['epoch'], df_yolo_s['val/box_loss'], label='val box', color='#3498db', linestyle='--')
    axes[2].plot(df_yolo_s['epoch'], df_yolo_s['train/cls_loss'], label='train cls', color='#e74c3c')
    axes[2].plot(df_yolo_s['epoch'], df_yolo_s['val/cls_loss'], label='val cls', color='#e74c3c', linestyle='--')
    axes[2].set_xlabel('Эпоха')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Losses (box / cls)')
    axes[2].legend(fontsize=8)

    plt.suptitle('Обучение YOLOv8s на исходных данных (349 изображений)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "02_yolov8s_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {REPORT_DIR / '02_yolov8s_training.png'}")
else:
    print(f"   ⚠️  Файл не найден: {results_path_1}")

# ============================================================
# 3. ГРАФИКИ ОБУЧЕНИЯ YOLOv8m (аугментированные данные)
# ============================================================
print("\n📈 3/7: Кривые обучения YOLOv8m на аугментированных данных...")

results_path_2 = Path("runs/detect/runs/yolo_aug/plates_detector_m/results.csv")
if results_path_2.exists():
    df_yolo_m = pd.read_csv(results_path_2)
    df_yolo_m.columns = df_yolo_m.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df_yolo_m['epoch'], df_yolo_m['metrics/mAP50(B)'], 'o-', color='#27ae60', linewidth=2, markersize=4)
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('mAP@0.5')
    axes[0].set_title('YOLOv8m: mAP@0.5 (val)')
    axes[0].axhline(0.80, color='red', linestyle='--', alpha=0.5, label='Цель 80%')
    axes[0].axhline(0.982, color='gold', linestyle=':', linewidth=2, label='Достигнуто: 98.2%')
    axes[0].legend()

    axes[1].plot(
        df_yolo_m['epoch'], df_yolo_m['metrics/precision(B)'], 'o-', label='Precision', color='#3498db', linewidth=2
    )
    axes[1].plot(df_yolo_m['epoch'], df_yolo_m['metrics/recall(B)'], 'o-', label='Recall', color='#e74c3c', linewidth=2)
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Значение')
    axes[1].set_title('Precision / Recall')
    axes[1].legend()

    axes[2].plot(df_yolo_m['epoch'], df_yolo_m['train/box_loss'], label='train box', color='#3498db')
    axes[2].plot(df_yolo_m['epoch'], df_yolo_m['val/box_loss'], label='val box', color='#3498db', linestyle='--')
    axes[2].plot(df_yolo_m['epoch'], df_yolo_m['train/cls_loss'], label='train cls', color='#e74c3c')
    axes[2].plot(df_yolo_m['epoch'], df_yolo_m['val/cls_loss'], label='val cls', color='#e74c3c', linestyle='--')
    axes[2].set_xlabel('Эпоха')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Losses (box / cls)')
    axes[2].legend(fontsize=8)

    plt.suptitle(
        'Обучение YOLOv8m на аугментированных данных (1381 изображение)', fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "03_yolov8m_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {REPORT_DIR / '03_yolov8m_training.png'}")

# ============================================================
# 4. ГРАФИКИ ОБУЧЕНИЯ FCOS (кастомная архитектура)
# ============================================================
print("\n📈 4/7: Кривые обучения FCOS...")

fcos_history = Path("weights/history.json")
if fcos_history.exists():
    with open(fcos_history) as f:
        history = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#3498db', linewidth=2)
    ax.plot(epochs, history['val_loss'], 's-', label='Val Loss', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Эпоха', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Кастомная FCOS: кривые обучения (1381 изображение)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "04_fcos_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {REPORT_DIR / '04_fcos_training.png'}")
else:
    print(f"   ⚠️  История FCOS не найдена: {fcos_history}")

# ============================================================
# 5. СРАВНИТЕЛЬНАЯ ТАБЛИЦА МОДЕЛЕЙ
# ============================================================
print("\n📊 5/7: Сравнительная таблица...")

comparison_data = {
    'Модель': ['FCOS (349 img)', 'FCOS (1381 img)', 'YOLOv8s (349 img)', 'YOLOv8m (1381 img)'],
    'Параметры': ['17.9M', '17.9M', '11.1M', '25.8M'],
    'Обуч. выборка': [349, 1381, 349, 1381],
    'mAP50 (val)': ['~60%', '—', '96.9%', '98.2%'],
    'Exact Match': ['~20%', '65%', '75%', '85%'],
    'Скорость (ms)': ['~15', '~15', '~10.5', '~9.5'],
    'Статус': ['❌ Не достигла', '⚠️ Близко к цели', '⚠️ Близко к цели', '✅ Цель достигнута'],
}
df_comp = pd.DataFrame(comparison_data)

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

table = ax.table(
    cellText=df_comp.values,
    colLabels=df_comp.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.22, 0.12, 0.14, 0.14, 0.14, 0.12, 0.18],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Раскраска заголовка
for j in range(len(df_comp.columns)):
    table[0, j].set_facecolor('#34495e')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Раскраска строки победителя
for j in range(len(df_comp.columns)):
    table[4, j].set_facecolor('#d5f4e6')
    table[4, j].set_text_props(fontweight='bold')

plt.title('Сравнение архитектур и подходов', fontsize=14, fontweight='bold', pad=20)
plt.savefig(REPORT_DIR / "05_comparison_table.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Сохранено: {REPORT_DIR / '05_comparison_table.png'}")

# ============================================================
# 6. ПРИМЕРЫ ПРЕДСКАЗАНИЙ YOLOv8m (успехи и ошибки)
# ============================================================
print("\n🖼️  6/7: Примеры предсказаний YOLOv8m...")

model = YOLO("runs/detect/runs/yolo_aug/plates_detector_m/weights/best.pt")
test_dir = Path("yolo_dataset_augmented/images/test")
labels_dir = Path("yolo_dataset_augmented/labels/test")

success_examples = []
fail_examples = []

for img_path in sorted(test_dir.glob("*.jpeg")):
    stem = img_path.stem
    txt_path = labels_dir / f"{stem}.txt"
    gt_str = get_gt_from_txt(txt_path)

    results = model(img_path, conf=0.25, iou=0.45, verbose=False)

    preds = []
    if len(results[0].boxes) > 0:
        for box, cls_id, conf in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
            strict=False,
        ):
            cx = (box[0] + box[2]) / 2
            preds.append((cx, CLASSES[int(cls_id)], box, conf))

    pred_chars = filter_plate_predictions(preds)
    pred_str = "".join(pred_chars)

    entry = {'path': img_path, 'gt': gt_str, 'pred': pred_str}
    if pred_str == gt_str and len(gt_str) > 0:
        success_examples.append(entry)
    else:
        fail_examples.append(entry)

# Рисуем 3 успеха и 3 ошибки
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for row, examples, title_prefix in [(0, success_examples[:3], "✅ Успех"), (1, fail_examples[:3], "❌ Ошибка")]:
    for col, ex in enumerate(examples):
        import matplotlib.image as mpimg

        img = mpimg.imread(ex['path'])
        axes[row, col].imshow(img)
        axes[row, col].set_title(
            f"{title_prefix}\nGT: {ex['gt']}  |  Pred: {ex['pred']}", fontsize=11, fontweight='bold'
        )
        axes[row, col].axis('off')

plt.suptitle('Примеры предсказаний YOLOv8m на тестовой выборке', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(REPORT_DIR / "06_prediction_examples.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Сохранено: {REPORT_DIR / '06_prediction_examples.png'}")

# ============================================================
# 7. АНАЛИЗ ОШИБОК (ТИПЫ ОШИБОК)
# ============================================================
print("\n🔍 7/7: Анализ типов ошибок...")

error_types = {'Пропуск символа': 0, 'Лишний символ (фантом)': 0, 'Подстановка символа': 0, 'Правильно': 0}

for ex in fail_examples:
    gt, pred = ex['gt'], ex['pred']
    # Простая эвристика классификации ошибки
    if len(pred) < len(gt):
        error_types['Пропуск символа'] += 1
    elif len(pred) > len(gt):
        error_types['Лишний символ (фантом)'] += 1
    else:
        error_types['Подстановка символа'] += 1

error_types['Правильно'] = len(success_examples)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#27ae60', '#e74c3c', '#f39c12', '#9b59b6']
wedges, texts, autotexts = ax.pie(
    error_types.values(),
    labels=error_types.keys(),
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 11},
)
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_color('white')

plt.title(
    f'Распределение результатов на тестовой выборке (20 изображений)\n'
    f'Всего: {len(success_examples)} правильно, {len(fail_examples)} ошибок',
    fontsize=14,
    fontweight='bold',
)
plt.savefig(REPORT_DIR / "07_error_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Сохранено: {REPORT_DIR / '07_error_analysis.png'}")

# ============================================================
# ФИНАЛЬНАЯ СВОДКА
# ============================================================
print("\n" + "=" * 60)
print("🎉 ВСЕ ВИЗУАЛИЗАЦИИ СГЕНЕРИРОВАНЫ!")
print("=" * 60)
print(f"\n📁 Папка с отчетом: {REPORT_DIR.absolute()}")
print("\nСодержимое:")
for f in sorted(REPORT_DIR.glob("*.png")):
    print(f"  • {f.name}")
print("\nТеперь создайте REPORT.md по шаблону ниже.")
