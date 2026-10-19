"""Тесты поиска изображений."""

from src.data.images import list_images


def test_lists_jpeg_and_png_sorted_without_annotations(tmp_path):
    for name in ['b.png', 'a.jpeg', 'c.JPG', 'a.txt', 'a.xml']:
        (tmp_path / name).touch()

    assert [p.name for p in list_images(tmp_path)] == ['a.jpeg', 'b.png', 'c.JPG']
