"""Тесты классов символов и чтения строки номера."""

from src.data.classes import LABEL_FIXES, read_label, read_plate


def test_read_plate_sorts_characters_left_to_right(tmp_path):
    txt = tmp_path / 'a.txt'
    txt.write_text('16 0.8 0.5 0.1 0.6\n5 0.1 0.5 0.1 0.6\n11 0.6 0.5 0.1 0.6\n7 0.3 0.5 0.1 0.6\n')

    assert read_plate(txt) == '57BJ'


def test_read_label_fixes_known_annotation_errors(tmp_path, monkeypatch):
    txt = tmp_path / 'bad.txt'
    txt.write_text('8 0.1 0.5 0.1 0.6\n0 0.3 0.5 0.1 0.6\n')
    monkeypatch.setitem(LABEL_FIXES, 'bad', '20')

    assert read_label(txt) == ['2 0.1 0.5 0.1 0.6', '0 0.3 0.5 0.1 0.6']
    assert read_plate(txt) == '20'
