"""Тесты offline-аугментации YOLO-датасета."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import augment_data


@pytest.fixture
def yolo_dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for split in ["train", "val", "test"]:
        (tmp_path / "yolo_dataset" / "images" / split).mkdir(parents=True)
        (tmp_path / "yolo_dataset" / "labels" / split).mkdir(parents=True)
    for name, labels in [
        ("a.jpeg", "3 0.3 0.5 0.1 0.6\n12 0.6 0.5 0.1 0.6\n"),
        # Бокс выходит за левый край - раньше такое изображение молча пропускалось
        ("b.png", "10 0.04 0.5 0.1 0.5\n"),
    ]:
        Image.fromarray(rng.integers(0, 255, (60, 120, 3), dtype=np.uint8)).save(
            tmp_path / "yolo_dataset" / "images" / "train" / name
        )
        (tmp_path / "yolo_dataset" / "labels" / "train" / name).with_suffix(".txt").write_text(labels)
    for split in ["val", "test"]:
        (tmp_path / "yolo_dataset" / "images" / split / "v.jpeg").touch()
        (tmp_path / "yolo_dataset" / "labels" / split / "v.txt").touch()
    monkeypatch.chdir(tmp_path)


def read_labels(directory):
    return {p.name: p.read_text() for p in sorted(directory.iterdir())}


def test_generates_three_augmentations_per_image(yolo_dataset):
    augment_data.augment_dataset()

    images = sorted(p.name for p in augment_data.OUT_IMAGES.iterdir())
    assert len(images) == 8
    assert {"a.jpeg", "b.png", "b_aug_0.jpeg"} <= set(images)
    assert len(list(augment_data.OUT_LABELS.iterdir())) == 8


def test_boxes_stay_normalized_and_originals_are_copied(yolo_dataset):
    augment_data.augment_dataset()

    for text in read_labels(augment_data.OUT_LABELS).values():
        for line in text.splitlines():
            assert all(0 <= float(v) <= 1 for v in line.split()[1:])
    original = Path("yolo_dataset/images/train/a.jpeg").read_bytes()
    assert (augment_data.OUT_IMAGES / "a.jpeg").read_bytes() == original


def test_val_and_test_are_copied_unchanged(yolo_dataset):
    augment_data.augment_dataset()

    for split in ["val", "test"]:
        assert [p.name for p in (augment_data.OUT_ROOT / "images" / split).iterdir()] == ["v.jpeg"]


def test_rerun_is_deterministic_and_removes_stale_files(yolo_dataset):
    augment_data.augment_dataset()
    first = read_labels(augment_data.OUT_LABELS)
    (augment_data.OUT_IMAGES / "stale.jpeg").touch()

    augment_data.augment_dataset()

    assert read_labels(augment_data.OUT_LABELS) == first
    assert not (augment_data.OUT_IMAGES / "stale.jpeg").exists()
