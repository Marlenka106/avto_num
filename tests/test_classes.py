"""Тесты классов символов и чтения строки номера."""

from pathlib import Path

import yaml

from src.data.classes import CLASSES, read_plate


def test_classes_match_yolo_config():
    config = yaml.safe_load((Path(__file__).parents[1] / 'data.yaml').read_text())

    assert config['names'] == CLASSES


def test_read_plate_sorts_characters_left_to_right(tmp_path):
    txt = tmp_path / 'a.txt'
    txt.write_text('16 0.8 0.5 0.1 0.6\n5 0.1 0.5 0.1 0.6\n11 0.6 0.5 0.1 0.6\n7 0.3 0.5 0.1 0.6\n')

    assert read_plate(txt) == '57BJ'
