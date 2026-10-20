"""Тесты метрик распознавания номеров."""

import pytest

from src.evaluation.metrics import cer, levenshtein, wilson_interval


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("5732NXD", "5732NXD", 0), ("5732NXD", "5782NXD", 1), ("5732NXD", "573NXD", 1), ("", "AB", 2), ("JB", "", 2)],
)
def test_levenshtein(a, b, expected):
    assert levenshtein(a, b) == expected


def test_cer_is_relative_to_ground_truth_length():
    assert cer("5732NXD", "5782ND") == pytest.approx(2 / 7)


def test_wilson_interval_matches_reference_values():
    low, high = wilson_interval(17, 20)

    assert low == pytest.approx(0.640, abs=1e-3)
    assert high == pytest.approx(0.948, abs=1e-3)


def test_wilson_interval_for_perfect_score_stays_below_one():
    low, high = wilson_interval(20, 20)

    assert 0.8 < low < 1
    assert high == pytest.approx(1.0)


def test_wilson_interval_for_zero_score_starts_at_zero():
    low, high = wilson_interval(0, 20)

    assert low == 0.0
    assert 0 < high < 0.2
