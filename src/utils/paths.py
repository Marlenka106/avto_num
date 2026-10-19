"""Централизованное управление путями в проекте."""
from pathlib import Path

# Корень проекта (на 2 уровня выше текущего файла: src/utils/paths.py -> project_root)
ROOT = Path(__file__).resolve().parents[2]

# Пути к данным
DATA_RAW = ROOT / "number_plates"
DATA_YOLO_AUG = ROOT / "yolo_dataset_augmented"

# Пути к артефактам
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"

# Автоматическое создание директорий при импорте
for p in [WEIGHTS_DIR, RUNS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)