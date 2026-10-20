# Параметры оценки: подбор порога - make eval-yolo SPLIT=val CONF="0.15 0.25 0.4"
YOLO_RUN ?= runs/yolo_aug/plates_detector_m
YOLO_WEIGHTS ?= $(YOLO_RUN)/weights/best.pt
FCOS_WEIGHTS ?= weights/best.pt
SPLIT ?= test
CONF ?= 0.25
# Демонстрация: make demo IMAGES=путь/к/папке_или_файлу
IMAGES ?= number_plates/test

.PHONY: install lint format test check eda split prepare augment data train-yolo eval-yolo train-fcos eval-fcos demo report clean

install:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

check: lint test

eda:
	python explore_data.py

split:
	python split_data.py

prepare:
	python prepare_yolo_data.py

augment:
	python augment_data.py

data: prepare augment

# Прошлый прогон не перезаписываем: его нужно перенести, иначе оценка возьмёт старые веса
train-yolo:
	@test ! -e $(YOLO_RUN) || (echo "$(YOLO_RUN) уже есть - перенесите прошлый прогон" && exit 1)
	yolo cfg=configs/yolo_train.yaml project=$(dir $(YOLO_RUN)) name=$(notdir $(YOLO_RUN))

eval-yolo:
	python evaluate.py --model yolo --weights $(YOLO_WEIGHTS) --split $(SPLIT) --conf $(CONF)

train-fcos:
	python -m src.training.train

eval-fcos:
	python evaluate.py --model fcos --weights $(FCOS_WEIGHTS) --split $(SPLIT) --conf $(CONF)

demo:
	python demo.py --model yolo --weights $(YOLO_WEIGHTS) --conf $(CONF) $(IMAGES)

report:
	python generate_report.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf runs/ weights/*.pt reports/final/*.png
