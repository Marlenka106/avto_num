"""Поиск изображений датасета."""

from pathlib import Path

# В датасете есть и .jpeg, и .png изображения
IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png'}


def list_images(directory):
    """Возвращает отсортированный список изображений в папке."""
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
