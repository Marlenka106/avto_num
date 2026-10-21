"""Генерация всех визуализаций для финального отчета."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ultralytics import YOLO

from evaluate import evaluate, make_yolo_predictor
from src.data.images import list_images
from src.data.quality import is_clean_annotation

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
# 1. ГРАФИКИ ОБУЧЕНИЯ YOLOv8s (первая модель)
# ============================================================
print("\n📈 1/4: Кривые обучения YOLOv8s...")

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

    plt.suptitle('Обучение YOLOv8s на исходных данных', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "02_yolov8s_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {REPORT_DIR / '02_yolov8s_training.png'}")
else:
    print(f"   ⚠️  Файл не найден: {results_path_1}")

# ============================================================
# 2. ГРАФИКИ ОБУЧЕНИЯ YOLOv8m (аугментированные данные)
# ============================================================
print("\n📈 2/4: Кривые обучения YOLOv8m на аугментированных данных...")

results_path_2 = Path("runs/yolo_aug/plates_detector_m/results.csv")
if results_path_2.exists():
    df_yolo_m = pd.read_csv(results_path_2)
    df_yolo_m.columns = df_yolo_m.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df_yolo_m['epoch'], df_yolo_m['metrics/mAP50(B)'], 'o-', color='#27ae60', linewidth=2, markersize=4)
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('mAP@0.5')
    axes[0].set_title('YOLOv8m: mAP@0.5 (val)')
    axes[0].axhline(0.80, color='red', linestyle='--', alpha=0.5, label='Цель 80%')
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

    plt.suptitle('Обучение YOLOv8m на аугментированных данных', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "03_yolov8m_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {REPORT_DIR / '03_yolov8m_training.png'}")

# ============================================================
# 3. ГРАФИКИ ОБУЧЕНИЯ FCOS (кастомная архитектура)
# ============================================================
print("\n📈 3/4: Кривые обучения FCOS...")

fcos_history = Path("weights/history.json")
if fcos_history.exists():
    with open(fcos_history) as f:
        history = json.load(f)

    fig, (ax, ax_em) = plt.subplots(1, 2, figsize=(16, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#3498db', linewidth=2)
    ax.plot(epochs, history['val_loss'], 's-', label='Val Loss', color='#e74c3c', linewidth=2)
    ax.set_xlabel('Эпоха', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Кастомная FCOS: кривые обучения', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # По Exact Match на val выбирается чекпоинт
    ax_em.plot(epochs, history['val_exact_match'], 'o-', color='#27ae60', linewidth=2)
    ax_em.axhline(y=0.8, color='red', linestyle='--', alpha=0.6, label='Цель 80%')
    ax_em.set_xlabel('Эпоха', fontsize=12)
    ax_em.set_ylabel('Exact Match', fontsize=12)
    ax_em.set_title('Кастомная FCOS: Exact Match (val)', fontsize=14, fontweight='bold')
    ax_em.legend()
    ax_em.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "04_fcos_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {REPORT_DIR / '04_fcos_training.png'}")
else:
    print(f"   ⚠️  История FCOS не найдена: {fcos_history}")

# ============================================================
# 4. ПРИМЕРЫ ПРЕДСКАЗАНИЙ YOLOv8m (успехи и ошибки)
# ============================================================
print("\n🖼️  4/4: Примеры предсказаний YOLOv8m...")

model = YOLO("runs/yolo_aug/plates_detector_m/weights/best.pt")
test_dir = Path("yolo_dataset_augmented/images/test")
labels_dir = Path("yolo_dataset_augmented/labels/test")

# Те же постобработка и порог, что в evaluate.py; неполная разметка не оценивается
images = [p for p in list_images(test_dir) if is_clean_annotation(labels_dir / f"{p.stem}.txt")]
rows = evaluate(images, labels_dir, make_yolo_predictor(model, conf=0.25), train_plates=set())

success_examples = []
fail_examples = []
for row in rows:
    entry = {'path': test_dir / row['name'], 'gt': row['gt'], 'pred': row['pred']}
    if row['pred'] == row['gt']:
        success_examples.append(entry)
    else:
        fail_examples.append(entry)

# Рисуем до 3 ошибок, остальные ячейки - успехи
fails = [(ex, "Ошибка") for ex in fail_examples[:3]]
examples = [(ex, "Успех") for ex in success_examples[: 6 - len(fails)]] + fails
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# Изображений может быть меньше шести - пустые ячейки не рисуем
for ax in axes.flat:
    ax.axis('off')

for ax, (ex, title_prefix) in zip(axes.flat, examples, strict=False):
    import matplotlib.image as mpimg

    img = mpimg.imread(ex['path'])
    ax.imshow(img)
    ax.set_title(f"{title_prefix}\nGT: {ex['gt']}  |  Pred: {ex['pred']}", fontsize=11, fontweight='bold')

plt.suptitle('Примеры предсказаний YOLOv8m на тестовой выборке', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(REPORT_DIR / "06_prediction_examples.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Сохранено: {REPORT_DIR / '06_prediction_examples.png'}")

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
