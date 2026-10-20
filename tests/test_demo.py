"""Тесты демонстрации (без модели: предсказания подставляются вручную)."""

from PIL import Image

import demo


def write_plate(labels_dir, stem, class_ids):
    lines = [f"{class_id} {0.1 * (i + 1):.2f} 0.5 0.05 0.6" for i, class_id in enumerate(class_ids)]
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def predict(img_path):
    """Номер 1234AAA на любом изображении."""
    return [([10 * i, 0, 10 * i + 8, 50], class_id, 0.9) for i, class_id in enumerate([1, 2, 3, 4, 10, 10, 10])]


def test_collect_images_from_dirs_and_files(tmp_path):
    for name in ["b.png", "a.jpeg", "a.txt"]:
        (tmp_path / name).touch()
    single = tmp_path / "single.jpg"

    assert demo.collect_images([tmp_path, single]) == [tmp_path / "a.jpeg", tmp_path / "b.png", single]


def test_recognize_compares_with_label_when_present(tmp_path, capsys):
    for stem in ["a", "b", "c"]:
        Image.new("RGB", (8, 8)).save(tmp_path / f"{stem}.png")
    write_plate(tmp_path, "a", [1, 2, 3, 4, 10, 10, 10])
    write_plate(tmp_path, "b", [5, 2, 3, 4, 10, 10, 10])

    correct, labeled = demo.recognize([tmp_path / f"{stem}.png" for stem in ["a", "b", "c"]], predict)

    assert (correct, labeled) == (1, 2)
    out = capsys.readouterr().out
    assert "✅ a.png" in out
    assert "GT: 5234AAA" in out
    assert "c.png" in out
