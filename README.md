# License Plate Character Detection & OCR

Профессиональный пайплайн для детекции и распознавания символов автомобильных номеров. В ходе исследования была разработана кастомная архитектура FCOS, а для достижения production-качества внедрен и дообучен YOLOv8.

**Итоговый результат:** **85.0% Exact Match Accuracy** на изолированной тестовой выборке (цель ≥ 80% достигнута ✅).

---

## Результаты экспериментов

| Модель | Обучающая выборка | mAP50 (val) | Exact Match (test) | Статус |
|---|---|---|---|---|
| FCOS (Custom) | 349 img | ~60% | ~20% | ❌ Baseline |
| YOLOv8s | 349 img | 96.9% | 75% | ⚠️ Близко к цели |
| **YOLOv8m** | **1381 img** (аугментировано) | **98.2%** | **85.0%** | **✅ Цель достигнута** |

*Скорость инференса финальной модели: ~9.5 мс на изображение (NVIDIA RTX 4060).*

---


## Архитектура проекта

avto_num/  
├── Makefile                  # Единая точка входа для всех команд  
├── pyproject.toml            # Современная конфигурация проекта  
├── requirements.txt          # Зависимости Python  
├── data.yaml                 # Конфигурация датасета для YOLO  
├── README.md                 # Главная страница репозитория  
├── REPORT.md                 # Полный текстовый отчет  
│  
├── augment_data.py           # Скрипт аугментации данных (Albumentations)  
├── evaluate_yolo_m.py        # Скрипт оценки финальной модели (YOLOv8m)  
├── evaluate_fcos_augmented.py# Скрипт оценки кастомной модели (FCOS)  
├── generate_report.py        # Скрипт генерации графиков для отчета  
│  
├── src/                      # Исходный код (модульная архитектура)  
│   ├── data/                 # Загрузка данных и трансформации  
│   │   ├── __init__.py  
│   │   └── dataset.py        # Кастомный PyTorch Dataset  
│   ├── models/               # Архитектуры нейросетей  
│   │   ├── __init__.py  
│   │   ├── fcos.py           # Кастомная архитектура FCOS + FPN  
│   │   └── loss.py           # Focal Loss + GIoU Loss  
│   ├── training/             # Циклы обучения  
│   │   ├── __init__.py  
│   │   └── train.py          # Training loop с gradient clipping  
│   ├── evaluation/           # Инференс и метрики  
│   │   ├── __init__.py  
│   │   └── inference.py      # Декодер и NMS для FCOS  
│   └── utils/                # Утилиты  
│       ├── __init__.py  
│       ├── paths.py          # Централизованное управление путями  
│       └── seed.py           # Фиксация seed для воспроизводимости  
│  
├── .gitignore                # Исключает тяжелые файлы из Git  
│  
# --- Директории, игнорируемые в Git (.gitignore), но присутствующие локально: ---  
├── .venv/                    # Виртуальное окружение Python  
├── number_plates/            # Исходные сырые данные (456 изображений)  
├── yolo_dataset_augmented/   # Аугментированный датасет (1381 изображение)  
├── runs/                     # Артефакты и логи обучения YOLO  
├── weights/                  # Сохраненные веса модели FCOS  
└── reports/                  # Сгенерированные графики и визуализации  


## Быстрый старт(Quick Start)  

Проект использует `Makefile` как единую точку входа для удобства воспроизведения результатов.  

```bash
# 1. Клонирование репозитория
git clone https://github.com/Marlenka106/avto_num.git
cd avto_num

# 2. Создание окружения и установка зависимостей
make install

# 3. Аугментация данных (генерирует ~1381 изображение из 349 исходных)
make augment

# 4. Обучение финальной модели (YOLOv8m)
make train-yolo

# 5. Оценка точности на тестовой выборке
make eval-yolo

# 6. Генерация финального отчета с графиками
make report
```

### Детальная настройка окружения (если make install требует уточнений)  
Для корректной работы с GPU (CUDA 12.4) рекомендуется устанавливать PyTorch явно:  

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Методология 

1. Честное разделение данных: 80% train, 20% val (из исходного train), 20 test (строго изолированная выборка, не участвовавшая в обучении).  
2. Аугментация данных: Albumentations pipeline (ShiftScaleRotate, MotionBlur, RandomBrightnessContrast, GaussNoise, CoarseDropout) + встроенные Mosaic/MixUp в YOLO. Итог: увеличение датасета в 4 раза с применением геометрических искажений, шума и изменения освещения.  
3. Early Stopping: patience=25 эпох для предотвращения переобучения.  
4. Постобработка: Кастомный геометрический фильтр, отбрасывающий фантомные детекции на основе анализа медианной высоты и горизонтального выравнивания, с ограничением максимальной длины номера (≤7 символов).  

## Классы (27 символов)  

Цифры: 0-9  

Буквы: A, B, D, E, G, H, J, K, L, N, R, S, T, U, V, X, Z  

## Технологии 

- Deep Learning: PyTorch 2.5+ (CUDA 12.4), Ultralytics YOLOv8
- Data Processing: Albumentations, OpenCV, Pandas
- Visualization: Matplotlib, Seaborn
- Environment: Python 3.10, Ubuntu 24.04, NVIDIA RTX 4060 Laptop (8GB VRAM)

## Дальнейшее развитие  

- Сбор реальных hard-negative примеров (изображений, на которых модель ошиблась) для дообучения.
- Реализация двухэтапного пайплайна: YOLO (детекция) + CRNN/SVTR (распознавание символов по кропам) для устранения ошибок типа D↔B или J↔L.
- Ансамблирование моделей (YOLOv8m + YOLOv8l) для повышения робастности.
- Экспорт модели в форматы ONNX / TensorRT для деплоя на edge-устройства.


## Лицензия
Учебный проект.  

Статус: Рабочий production-ready прототип с точностью 85% (Exact Match).

Детальный анализ, графики обучения и разбор ошибок доступны в файле REPORT.md и в папке reports/final/.



