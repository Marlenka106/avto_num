"""Воспроизводимость онлайн-аугментаций FCOS, включая процессы DataLoader."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import PlateCharDataset, get_transform
from src.training.train import collate_fn, seed_worker
from src.utils.seed import set_seed


@pytest.fixture
def repeated_images(tmp_path):
    # Одинаковые картинки и боксы: различия должны появляться только из-за аугментаций.
    image = np.random.default_rng(0).integers(0, 256, (64, 128, 3), dtype=np.uint8)
    for i in range(8):
        Image.fromarray(image).save(tmp_path / f'{i}.png')
        # Разные классы позволяют также проверить воспроизводимость перемешивания.
        (tmp_path / f'{i}.txt').write_text(f'{i} 0.5 0.5 0.2 0.6\n')
    return tmp_path


def make_loader(directory, seed, num_workers, shuffle=False):
    set_seed(seed)
    dataset = PlateCharDataset(directory, directory, get_transform(train=True, seed=seed))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def snapshot(loader):
    return [(images, targets[0]['boxes'], targets[0]['labels']) for images, targets in loader]


def same_epoch(first, second):
    return len(first) == len(second) and all(
        torch.equal(a, b)
        for first_sample, second_sample in zip(first, second, strict=True)
        for a, b in zip(first_sample, second_sample, strict=True)
    )


@pytest.mark.parametrize('num_workers', [0, 4])
def test_same_seed_reproduces_images_boxes_and_shuffle_across_epochs(repeated_images, num_workers):
    first_loader = make_loader(repeated_images, seed=42, num_workers=num_workers, shuffle=True)
    first_run = [snapshot(first_loader), snapshot(first_loader)]

    second_loader = make_loader(repeated_images, seed=42, num_workers=num_workers, shuffle=True)
    second_run = [snapshot(second_loader), snapshot(second_loader)]

    assert all(same_epoch(a, b) for a, b in zip(first_run, second_run, strict=True))


@pytest.mark.parametrize('num_workers', [0, 4])
def test_different_seed_changes_augmentations(repeated_images, num_workers):
    first = snapshot(make_loader(repeated_images, seed=42, num_workers=num_workers))
    second = snapshot(make_loader(repeated_images, seed=43, num_workers=num_workers))

    assert any(not torch.equal(a[0], b[0]) for a, b in zip(first, second, strict=True))


def test_workers_use_different_augmentation_streams(repeated_images):
    samples = snapshot(make_loader(repeated_images, seed=42, num_workers=4))

    # Первые четыре одинаковые картинки обрабатывают четыре разных worker.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.equal(samples[i][0], samples[j][0])


@pytest.mark.parametrize('num_workers', [0, 4])
def test_augmentations_change_between_epochs(repeated_images, num_workers):
    loader = make_loader(repeated_images, seed=42, num_workers=num_workers)

    first, second = snapshot(loader), snapshot(loader)

    # shuffle=False: сравниваем преобразования одних и тех же изображений.
    assert any(not torch.equal(a[0], b[0]) for a, b in zip(first, second, strict=True))
