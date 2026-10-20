"""Тесты проверки качества разметки."""

import pytest

from src.data.quality import is_clean_annotation


def write_labels(tmp_path, heights):
    txt = tmp_path / "a.txt"
    txt.write_text("".join(f"3 {0.1 * (i + 1):.2f} 0.5 0.05 {h}\n" for i, h in enumerate(heights)))
    return txt


def test_full_plate_is_clean(tmp_path):
    assert is_clean_annotation(write_labels(tmp_path, [0.6] * 7))


@pytest.mark.parametrize("n_boxes", [2, 6, 8])
def test_partial_or_extra_boxes_are_rejected(tmp_path, n_boxes):
    assert not is_clean_annotation(write_labels(tmp_path, [0.6] * n_boxes))


def test_degenerate_box_is_rejected(tmp_path):
    assert not is_clean_annotation(write_labels(tmp_path, [0.6] * 6 + [0.02]))
