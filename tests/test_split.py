"""Тесты разделения выборки и подготовки YOLO-датасета."""

from pathlib import Path

import pytest

import prepare_yolo_data
import split_data


def make_image(directory, name, plate):
    """Создаёт пустое изображение и YOLO-разметку номера (class_id слева направо)."""
    (directory / name).touch()
    lines = [f"{class_id} {0.1 * (i + 1):.2f} 0.5 0.05 0.6" for i, class_id in enumerate(plate)]
    # Строки в .txt не всегда идут слева направо
    (directory / name).with_suffix(".txt").write_text("\n".join(reversed(lines)) + "\n")
    (directory / name).with_suffix(".xml").touch()
    return directory / name


@pytest.fixture
def images(tmp_path):
    plates = [[i % 10, (i // 10) % 10, 15] for i in range(20)]
    # Один номер снят 5 раз
    repeated = [make_image(tmp_path, f"same_{i}.jpeg", [5, 7, 3, 2]) for i in range(5)]
    unique = [make_image(tmp_path, f"img_{i:02d}.png", plate) for i, plate in enumerate(plates)]
    return sorted(repeated + unique)


def test_plate_key_is_left_to_right(tmp_path):
    img = make_image(tmp_path, "a.jpeg", [5, 7, 3, 2, 18])

    assert split_data.plate_key(img.with_suffix(".txt")) == "5 7 3 2 18"


def test_same_plate_never_in_both_splits(images):
    train, val = split_data.split_by_plate(images)

    train_plates = {split_data.plate_key(p.with_suffix(".txt")) for p in train}
    val_plates = {split_data.plate_key(p.with_suffix(".txt")) for p in val}
    assert not train_plates & val_plates
    assert sorted(train + val) == images


def test_split_is_deterministic_and_close_to_fraction(images):
    first = split_data.split_by_plate(images)

    assert split_data.split_by_plate(list(reversed(images))) == first
    assert 0.15 <= len(first[1]) / len(images) <= 0.4


def test_pipeline_writes_splits_and_builds_yolo_dataset(tmp_path, monkeypatch):
    for split in ["train", "test"]:
        (tmp_path / "number_plates" / split).mkdir(parents=True)
    train_images = [make_image(tmp_path / "number_plates" / "train", f"t{i}.jpeg", [i, 1, 2]) for i in range(10)]
    make_image(tmp_path / "number_plates" / "test", "x.png", [4, 4, 4])
    monkeypatch.chdir(tmp_path)

    split_data.main()
    prepare_yolo_data.prepare_yolo_dataset()

    train_names = Path("splits/train.txt").read_text().split()
    val_names = Path("splits/val.txt").read_text().split()
    assert sorted(train_names + val_names) == [p.name for p in train_images]
    assert sorted(p.name for p in Path("yolo_dataset/images/val").iterdir()) == sorted(val_names)
    assert len(list(Path("yolo_dataset/labels/train").iterdir())) == len(train_names)
    assert [p.name for p in Path("yolo_dataset/images/test").iterdir()] == ["x.png"]
    assert len(list(Path("number_plates/train").iterdir())) == 30
