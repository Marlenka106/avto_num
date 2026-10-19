# 🚗 License Plate Character Detection & OCR

Детектор символов номерных знаков с использованием кастомной архитектуры FCOS и YOLOv8.

## Результаты

### YOLOv8 (Production-ready)
- **mAP50: 96.9%** (валидация)
- **Exact Match Accuracy: 75%** (тестовая выборка, 20 изображений)
- **Скорость инференса: ~10ms** на RTX 4060

### FCOS (Кастомная архитектура)
- mAP50: ~60% (недообучена из-за малого датасета)
- Full pipeline: backbone → FPN → FCOS head → NMS

## Архитектура проекта
avto_num/  
├── src/  
│ ├── data/                   # Загрузка данных, аугментации  
│ │ ├── dataset.py           # PyTorch Dataset  
│ │ ├── loader.py           # Парсинг XML/TXT аннотаций  
│ │ └── eda.py           # Разведочный анализ  
│ ├── models/           # Архитектуры моделей  
│ │ ├── fcos.py          # FCOS detector  
│ │ └── loss.py           # Focal + GIoU loss  
│ ├── training/          # Циклы обучения  
│ │ └── train.py           # Training loop  
│ ├── evaluation/         # Инференс и метрики  
│ │ ├── inference.py       # FCOS inference  
│ │ └── metrics.py         # mAP, IoU  
│ └── utils/             # Утилиты  
│ ├── seed.py           # Reproducibility  
│ └── paths.py           # Paths management  
├── number_plates/       # Исходные данные (train/val/test)  
├── yolo_dataset/         # Данные в формате YOLO  
├── runs/                 # Логи и веса YOLO  
├── weights/             # Веса FCOS  
├── reports/           # Графики и визуализации  
├── evaluate_yolo.py       # Финальная оценка YOLO  
├── prepare_yolo_data.py       # Конвертация в YOLO формат  
├── data.yaml               # Конфиг YOLO  
└── README.md  


## Быстрый старт

### Установка зависимостей
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics opencv-python-headless pandas matplotlib seaborn scikit-learn tqdm Pillow pyyaml loguru albumentations
```

### Обучение YOLOv8 

```bash
yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=8 patience=20
```

### Оценка точности

```bash
python evaluate_yolo.py
```
## Методология 

1. Разделение данных: 80% train, 20% val (из исходного train), 20 test (изолированная выборка)
2. Аугментации: Mosaic, MixUp, ColorJitter, RandomAffine
3. Early Stopping: patience=20 эпох
4. Постобработка: геометрическая фильтрация + ограничение длины номера (≤7 символов)  

## Классы (27 символов)  

Цифры: 0-9  

Буквы: A, B, D, E, G, H, J, K, L, N, R, S, T, U, V, X, Z  

## TODO  

- Увеличить датасет (аугментация существующих данных)
- Попробовать YOLOv8m/l для лучшей точности
- Добавить OCR-постобработку (CRNN/SVTR)
- Ensemble из нескольких моделей
- Оптимизация для edge-устройств (TensorRT, ONNX)  

## Технологии 

- PyTorch 2.5+ с CUDA 12.4
- Ultralytics YOLOv8 для production-детекции
- Кастомная FCOS для исследовательских целей
- Python 3.10
- Ubuntu 24.04 + RTX 4060  

## Лицензия
Учебный проект.  

Статус: Рабочий прототип с точностью 75% (Exact Match)  

Цель: Довести до 80%+ через аугментацию и более мощные модели

## Финальные результаты (После аугментации и YOLOv8m)

- **mAP50:** 98.2%
- **Exact Match Accuracy:** 85.0% (17/20 на изолированном тесте)
- **Размер обучающей выборки:** 1381 изображение (аугментировано из 349)
- **Модель:** YOLOv8m (25.8M параметров)

### Как повторить результат:
1. Сгенерировать аугментированный датасет: `python augment_data.py`
2. Обучить модель: `yolo detect train data=data.yaml model=yolov8m.pt epochs=100 imgsz=640 batch=8 patience=25 project=runs/yolo_aug name=plates_detector_m`
3. Оценить точность: `python evaluate_yolo_m.py`

