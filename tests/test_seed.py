"""Тесты утилит воспроизводимости."""

import random

import numpy as np
import torch

from src.utils.seed import set_seed


def _draw():
    return random.random(), np.random.rand(), torch.rand(1).item()


def test_same_seed_gives_same_random_streams():
    set_seed(123)
    first = _draw()
    set_seed(123)

    assert _draw() == first


def test_different_seeds_give_different_streams():
    set_seed(1)
    first = _draw()
    set_seed(2)

    assert _draw() != first


def test_enables_deterministic_cudnn():
    set_seed(0)

    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
