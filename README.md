# Распознавание символов на автомобильных номерных знаках

Пайплайн для детекции и распознавания символов саудовских номерных знаков. В ходе исследования была разработана кастомная архитектура FCOS, а как основная модель используется YOLOv8.

## Результаты

Обе модели обучены по новому пайплайну, порог уверенности подобран на val (YOLOv8m - 0.25, FCOS - 0.3), test запускался один раз. Exact Match, в скобках - 95% доверительный интервал:

| Выборка | Изображений | YOLOv8m | FCOS |
|---|---|---|---|
| val (номеров нет в train) | 94 | 91.5% [84.1%; 95.6%] | 90.4% [82.8%; 94.9%] |
| test, все | 26 | 96.2% (25/26) [81.1%; 99.3%] | 88.5% (23/26) [71.0%; 96.0%] |
| test, номер есть в train | 20 | 95.0% (19/20) | 90.0% (18/20) |
| test, новый номер | 6 | 100% (6/6) | 83.3% (5/6) |
| CER на test | | 0.5% | 1.6% |

Цель 80% Exact Match достигнута обеими моделями; основная модель - YOLOv8m. На val модели почти равны, но у FCOS по Exact Match на val выбирались и чекпоинт, и порог, поэтому её цифра на val немного завышена. Разбор ошибок - в [REPORT.md](REPORT.md).

Прежние результаты (85% Exact Match у YOLOv8m, ~20% и 65% у FCOS) получены по старому протоколу и недействительны: test был урезан до 20 изображений (PNG не учитывались), в нём были изображения с неполной разметкой, номера test встречаются в train, а порог уверенности подбирался на тесте. Подробности - в [REPORT.md](REPORT.md).

## Данные

593 кропа номерных знаков (train 563 / test 30, по одному номеру на изображение), разметка символов в YOLO `.txt` и Pascal VOC `.xml`, 27 классов. Ключевые особенности (полный анализ - [reports/eda.md](reports/eda.md), `make eda`):

- около 23% изображений - PNG;
- у 109 изображений train и 4 test разметка неполная (меньше 7 символов);
- один автомобиль снят многократно: 286 уникальных номеров на 563 изображения train, 23 из 30 номеров test есть в train;
- в test нет ни одного `T`, у многих букв 1-3 примера;
- `.xml` и `.txt` согласованы; 4 ошибки разметки в train исправлены при чтении (`LABEL_FIXES`).

## Структура проекта

```
avto_num/
├── Makefile                  # Единая точка входа для всех команд
├── pyproject.toml            # Настройки ruff и pytest
├── requirements.txt          # Зависимости Python (версии зафиксированы)
├── requirements-dev.txt      # Зависимости для разработки (ruff, pytest, pre-commit)
├── configs/yolo_train.yaml   # Гиперпараметры обучения YOLO (без отражений, seed)
├── splits/                   # Списки train/val: деление по номеру, только полная разметка
├── README.md
├── REPORT.md                 # Отчёт: подход, история экспериментов, статус
│
├── explore_data.py           # EDA: статистика и графики в reports/figures
├── split_data.py             # Создание splits/*.txt
├── prepare_yolo_data.py      # Сборка yolo_dataset по спискам и генерация data.yaml
├── augment_data.py           # Offline-аугментация train (Albumentations)
├── evaluate.py               # Оценка YOLO и FCOS: Exact Match, CER, 95% ДИ
├── generate_report.py        # Графики обучения и примеры предсказаний для отчёта
│
├── src/
│   ├── data/
│   │   ├── classes.py        # CLASSES и чтение строки номера
│   │   ├── dataset.py        # PyTorch Dataset для FCOS
│   │   ├── images.py         # Поиск изображений (.jpeg, .jpg, .png)
│   │   ├── quality.py        # Проверка полноты разметки
│   │   └── yolo_config.py    # Генерация data.yaml
│   ├── models/
│   │   ├── fcos.py           # Кастомная архитектура FCOS + FPN
│   │   └── loss.py           # Focal Loss + GIoU Loss, назначение по уровням FPN
│   ├── training/train.py     # Обучение FCOS, выбор чекпоинта по Exact Match на val
│   ├── evaluation/
│   │   ├── inference.py      # Декодер и NMS для FCOS
│   │   ├── metrics.py        # Exact Match, CER, интервал Уилсона
│   │   └── postprocess.py    # Геометрический фильтр символов номера
│   └── utils/seed.py         # Фиксация seed для воспроизводимости
├── tests/                    # pytest
└── reports/
    ├── eda.md                # Разведочный анализ с выводами
    ├── figures/              # Графики EDA
    └── final/                # Графики отчёта (generate_report.py)

# Не хранятся в git:
├── number_plates/            # Исходные данные: train/ и test/
├── yolo_dataset/             # Датасет в формате YOLO (make prepare)
├── yolo_dataset_augmented/   # Аугментированный датасет, 1420 изображений train (make augment)
├── runs/                     # Логи и веса YOLO
└── weights/                  # Веса и история обучения FCOS
```

## Быстрый старт (Quick Start)

Проект использует `Makefile` как единую точку входа. Исходные данные нужно положить в `number_plates/` (папки `train/` и `test/`).

```bash
# 1. Клонирование репозитория
git clone https://github.com/Marlenka106/avto_num.git
cd avto_num

# 2. Установка зависимостей
make install

# 3. Разведочный анализ (графики в reports/figures)
make eda

# 4. Подготовка YOLO-датасета по спискам splits/*.txt и аугментация (исходные данные не изменяются)
make data

# 5. Обучение YOLOv8m (гиперпараметры в configs/yolo_train.yaml)
make train-yolo

# 6. Подбор порога уверенности на val и финальная оценка на test
make eval-yolo SPLIT=val CONF="0.15 0.25 0.4"
make eval-yolo SPLIT=test CONF=0.25

# 7. Графики для отчёта
make report
```

FCOS: `make train-fcos` (веса в `weights/best.pt`), затем `make eval-fcos SPLIT=val CONF="0.2 0.3 0.4"` и `make eval-fcos SPLIT=test CONF=0.3`.

### Детальная настройка окружения

Для работы с GPU (CUDA 12.4) PyTorch устанавливается явно:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Разработка

```bash
pip install -r requirements-dev.txt
pre-commit install   # ruff + ruff format перед каждым коммитом

make lint            # ruff check + ruff format --check
make test            # pytest
make check           # всё сразу, как в CI
```

CI (GitHub Actions) запускает линтеры и тесты на Python 3.10 для каждого PR.

## Методология

1. Данные: учитываются изображения всех форматов (`.jpeg`, `.jpg`, `.png`); изображения с неполной разметкой (не 7 символов или боксы-«клики») не используются ни в обучении, ни в оценке.
2. Разделение: train/val 80/20 из исходного train группами по номеру - снимки одного номера не попадают одновременно в train и val, номера с больше чем 5 кадрами - только в train (списки в `splits/`, пересоздать - `make split`). Официальный test используется только для финальной оценки.
3. Аугментация: Albumentations (Affine, MotionBlur, GaussianBlur, RandomBrightnessContrast, HueSaturationValue, GaussNoise, ISONoise; seed 42) + встроенные аугментации YOLO (Mosaic, HSV, сдвиг и масштаб) без отражений: `fliplr: 0`, зеркальный символ - это другой глиф.
4. Постобработка: геометрический фильтр, отбрасывающий детекции с выбросами по высоте и по линии, и ограничение длины номера (до 7 самых уверенных символов). Общий для YOLO и FCOS.
5. Оценка: Exact Match с 95% доверительным интервалом Уилсона, CER и Exact Match по номерам (каждый номер с одинаковым весом); порог уверенности подбирается на val, test запускается один раз. Результат выводится отдельно для номеров, которые есть в train, и для новых номеров.

## Классы (27 символов)

Цифры: 0-9

Буквы: A, B, D, E, G, H, J, K, L, N, R, S, T, U, V, X, Z

## Технологии

- Deep Learning: PyTorch 2.5 (CUDA 12.4), Ultralytics YOLOv8
- Data Processing: Albumentations, OpenCV, Pandas
- Visualization: Matplotlib, Seaborn
- Environment: Python 3.10, Ubuntu 24.04, NVIDIA RTX 4060 Laptop (8GB VRAM)

## Дальнейшее развитие

- Учесть в геометрическом фильтре длинные однострочные номера, где буквы крупнее цифр (2 из 8 ошибок на val).
- Доразметить изображения с неполной разметкой вместо их исключения.
- Собрать test из номеров, которых нет в train, - сейчас обобщение на новые номера проверяется на 6 изображениях.
- Двухэтапный пайплайн: YOLO (детекция) + CRNN/SVTR (распознавание символов по кропам) для ошибок типа D↔B или J↔L.
- Экспорт модели в ONNX / TensorRT для деплоя на edge-устройства.

## Лицензия

Учебный проект.
