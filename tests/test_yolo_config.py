"""Тесты конфигурации обучения YOLO."""

from pathlib import Path

import yaml

from src.data.classes import CLASSES
from src.data.yolo_config import write_data_yaml

ROOT = Path(__file__).parents[1]


def test_data_yaml_has_absolute_path_and_project_classes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "yolo_dataset").mkdir()

    write_data_yaml(Path("yolo_dataset"))

    config = yaml.safe_load((tmp_path / "yolo_dataset" / "data.yaml").read_text())
    assert config["path"] == str(tmp_path / "yolo_dataset")
    assert config["names"] == CLASSES
    assert [config[split] for split in ["train", "val", "test"]] == ["images/train", "images/val", "images/test"]


def test_train_config_disables_flips_and_fixes_seed():
    config = yaml.safe_load((ROOT / "configs" / "yolo_train.yaml").read_text())

    assert config["fliplr"] == 0.0
    assert config["seed"] == 42
    assert config["data"] == "yolo_dataset_augmented/data.yaml"


def test_train_config_lists_augmentations_and_keeps_previous_runs():
    config = yaml.safe_load((ROOT / "configs" / "yolo_train.yaml").read_text())

    augmentations = ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective"]
    augmentations += ["flipud", "fliplr", "mosaic", "close_mosaic", "mixup", "copy_paste"]
    assert all(key in config for key in augmentations)
    assert config["mixup"] == 0.0
    assert config["exist_ok"] is False
