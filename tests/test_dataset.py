"""Тесты Dataset символов номерных знаков."""

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import INPUT_HEIGHT, INPUT_WIDTH, PlateCharDataset, get_transform
from src.training.train import collate_fn


@pytest.fixture
def data_dir(tmp_path):
    Image.new('RGB', (200, 100), 'white').save(tmp_path / 'a.jpeg')
    (tmp_path / 'a.txt').write_text('3 0.5 0.5 0.1 0.6\n12 0.7 0.5 0.1 0.6\n')
    # PNG с альфа-каналом и боксом, выходящим за левый край
    Image.new('RGBA', (100, 50), 'white').save(tmp_path / 'b.png')
    (tmp_path / 'b.txt').write_text('10 0.02 0.5 0.06 0.5\n')
    return tmp_path


def test_loads_jpeg_and_png(data_dir):
    dataset = PlateCharDataset(data_dir, data_dir, get_transform(train=False))

    assert [p.name for p in dataset.img_paths] == ['a.jpeg', 'b.png']
    assert len(dataset) == 2


def test_item_is_normalized_tensor_with_resized_boxes(data_dir):
    dataset = PlateCharDataset(data_dir, data_dir, get_transform(train=False))

    image, target = dataset[0]

    assert image.shape == (3, INPUT_HEIGHT, INPUT_WIDTH)
    assert image.dtype == torch.float32
    assert target['labels'].tolist() == [3, 12]
    expected = torch.tensor([[230.4, 51.2, 281.6, 204.8], [332.8, 51.2, 384.0, 204.8]])
    assert torch.allclose(target['boxes'], expected, atol=1e-3)


def test_boxes_outside_image_are_clipped(data_dir):
    dataset = PlateCharDataset(data_dir, data_dir, get_transform(train=False))

    image, target = dataset[1]

    assert image.shape[0] == 3
    assert torch.allclose(target['boxes'], torch.tensor([[0.0, 64.0, 25.6, 192.0]]), atol=1e-3)


def test_train_transform_keeps_boxes_inside_image(data_dir):
    torch.manual_seed(0)
    dataset = PlateCharDataset(data_dir, data_dir, get_transform(train=True))

    for _ in range(10):
        image, target = dataset[0]
        boxes = target['boxes']
        assert image.shape == (3, INPUT_HEIGHT, INPUT_WIDTH)
        assert len(boxes) == len(target['labels'])
        assert (boxes[:, [0, 2]] <= INPUT_WIDTH).all()
        assert (boxes[:, [1, 3]] <= INPUT_HEIGHT).all()
        assert (boxes >= 0).all()


def test_batches_with_train_collate_fn(data_dir):
    dataset = PlateCharDataset(data_dir, data_dir, get_transform(train=False))

    images, targets = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_fn)))

    assert images.shape == (2, 3, INPUT_HEIGHT, INPUT_WIDTH)
    assert [len(t['labels']) for t in targets] == [2, 1]
