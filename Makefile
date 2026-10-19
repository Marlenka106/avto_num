.PHONY: install lint format test check split prepare augment train-yolo eval-yolo train-fcos eval-fcos report clean

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

split:
	python split_data.py

prepare:
	python prepare_yolo_data.py

augment:
	python augment_data.py

train-yolo:
	yolo detect train data=data.yaml model=yolov8m.pt epochs=100 imgsz=640 batch=8 patience=25 project=runs/yolo_aug name=plates_detector_m

eval-yolo:
	python evaluate_yolo_m.py

train-fcos:
	python -m src.training.train

eval-fcos:
	python evaluate_fcos_augmented.py

report:
	python generate_report.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf runs/ weights/*.pt reports/final/*.png
