"""Генерация data.yaml для Ultralytics."""

import yaml

from src.data.classes import CLASSES


def write_data_yaml(root):
    """Пишет root/data.yaml с путями к сплитам и именами классов."""
    # Относительный path Ultralytics ищет в своей папке datasets, поэтому путь абсолютный
    config = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASSES,
    }
    (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
