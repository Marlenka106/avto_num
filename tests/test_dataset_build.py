"""Сохранность предыдущего датасета при ошибках подготовки и аугментации."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

import augment_data
import prepare_yolo_data
from src.data import build as dataset_build


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def source_data(tmp_path, monkeypatch):
    image = np.random.default_rng(0).integers(0, 256, (60, 120, 3), dtype=np.uint8)
    for split, names in [('train', ['a', 'b']), ('test', ['c'])]:
        directory = tmp_path / 'number_plates' / split
        directory.mkdir(parents=True)
        for name in names:
            Image.fromarray(image).save(directory / f'{name}.png')
            (directory / f'{name}.txt').write_text('3 0.3 0.5 0.1 0.6\n12 0.6 0.5 0.1 0.6\n')
    (tmp_path / 'splits').mkdir()
    (tmp_path / 'splits/train.txt').write_text('a.png\n')
    (tmp_path / 'splits/val.txt').write_text('b.png\n')
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(params=['prepare', 'augment'])
def builder(source_data, request):
    if request.param == 'prepare':
        return SimpleNamespace(
            run=prepare_yolo_data.prepare_yolo_dataset,
            output=Path('yolo_dataset'),
            missing=Path('splits/val.txt'),
            label=Path('number_plates/train/a.txt'),
        )
    prepare_yolo_data.prepare_yolo_dataset()
    return SimpleNamespace(
        run=augment_data.augment_dataset,
        output=Path('yolo_dataset_augmented'),
        missing=Path('yolo_dataset/images/val'),
        label=Path('yolo_dataset/labels/train/a.txt'),
    )


def test_missing_input_preserves_previous_dataset(builder):
    builder.run()
    before = snapshot(builder.output)
    builder.missing.rename(builder.missing.with_name(builder.missing.name + '.unavailable'))

    with pytest.raises(FileNotFoundError):
        builder.run()

    assert snapshot(builder.output) == before
    assert not list(Path('.').glob(f'.{builder.output.name}-*'))


def test_missing_label_preserves_previous_dataset(builder):
    builder.run()
    before = snapshot(builder.output)
    builder.label.unlink()

    with pytest.raises(FileNotFoundError):
        builder.run()

    assert snapshot(builder.output) == before


def test_successful_rebuild_replaces_stale_files_and_uses_final_yaml_path(builder):
    builder.run()
    before = snapshot(builder.output)
    (builder.output / 'stale.txt').write_text('old dataset')

    builder.run()

    assert snapshot(builder.output) == before
    config = yaml.safe_load((builder.output / 'data.yaml').read_text())
    assert config['path'] == str(builder.output.resolve())
    assert not list(Path('.').glob(f'.{builder.output.name}-*'))


def test_label_read_error_after_copy_preserves_previous_dataset(source_data, monkeypatch):
    prepare_yolo_data.prepare_yolo_dataset()
    output = Path('yolo_dataset')
    before = snapshot(output)
    read_label = prepare_yolo_data.read_label

    def fail_on_val(path):
        if path.stem == 'b':
            raise ValueError('invalid annotation')
        return read_label(path)

    monkeypatch.setattr(prepare_yolo_data, 'read_label', fail_on_val)
    with pytest.raises(ValueError, match='invalid annotation'):
        prepare_yolo_data.prepare_yolo_dataset()

    assert snapshot(output) == before


@pytest.mark.parametrize('raise_error', [False, True])
def test_image_write_failure_preserves_previous_dataset(source_data, monkeypatch, raise_error):
    prepare_yolo_data.prepare_yolo_dataset()
    augment_data.augment_dataset()
    output = Path('yolo_dataset_augmented')
    before = snapshot(output)
    imwrite = augment_data.cv2.imwrite

    def fail_second_image(path, image):
        if Path(path).stem == 'a_aug_1':
            if raise_error:
                raise OSError('image write failed')
            return False
        return imwrite(path, image)

    monkeypatch.setattr(augment_data.cv2, 'imwrite', fail_second_image)
    with pytest.raises(OSError):
        augment_data.augment_dataset()

    assert snapshot(output) == before


def test_yaml_write_failure_preserves_previous_dataset(builder, monkeypatch):
    builder.run()
    before = snapshot(builder.output)

    def fail_yaml(*args, **kwargs):
        raise OSError('yaml write failed')

    monkeypatch.setattr(dataset_build, 'write_data_yaml', fail_yaml)
    with pytest.raises(OSError, match='yaml write failed'):
        builder.run()

    assert snapshot(builder.output) == before
    assert not list(Path('.').glob(f'.{builder.output.name}-*'))


@pytest.mark.parametrize('phase', ['backup', 'publish'])
def test_replacement_failure_preserves_previous_dataset(builder, monkeypatch, phase):
    builder.run()
    before = snapshot(builder.output)
    rename = Path.rename

    def fail_rename(path, target):
        if (phase == 'backup' and path == builder.output) or (
            phase == 'publish' and path.name == 'new' and Path(target) == builder.output
        ):
            raise OSError('replacement failed')
        return rename(path, target)

    monkeypatch.setattr(Path, 'rename', fail_rename)
    with pytest.raises(OSError, match='replacement failed'):
        builder.run()

    assert snapshot(builder.output) == before
    assert not list(Path('.').glob(f'.{builder.output.name}-*'))


def test_failed_first_build_does_not_leave_partial_dataset(builder):
    builder.missing.rename(builder.missing.with_name(builder.missing.name + '.unavailable'))

    with pytest.raises(FileNotFoundError):
        builder.run()

    assert not builder.output.exists()
    assert not list(Path('.').glob(f'.{builder.output.name}-*'))


def test_failed_rollback_keeps_backup_and_reports_its_location(tmp_path, monkeypatch):
    output = tmp_path / 'dataset'
    output.mkdir()
    (output / 'original.txt').write_text('previous dataset')
    before = snapshot(output)
    rename = Path.rename

    def fail_publication_and_rollback(path, target):
        if path.name in {'new', 'previous'}:
            raise OSError('filesystem error')
        return rename(path, target)

    monkeypatch.setattr(Path, 'rename', fail_publication_and_rollback)
    with (
        pytest.raises(OSError, match='предыдущий датасет сохранён в') as error,
        dataset_build.staged_dataset(output) as staged,
    ):
        (staged / 'new.txt').write_text('new dataset')

    backups = list(tmp_path.glob('.dataset-*/previous'))
    assert len(backups) == 1
    assert str(backups[0]) in str(error.value)
    assert snapshot(backups[0]) == before


@pytest.mark.parametrize('broken_file', ['image', 'label'])
def test_invalid_training_data_preserves_previous_augmented_dataset(source_data, broken_file):
    prepare_yolo_data.prepare_yolo_dataset()
    augment_data.augment_dataset()
    output = Path('yolo_dataset_augmented')
    before = snapshot(output)
    if broken_file == 'image':
        Path('yolo_dataset/images/train/a.png').write_bytes(b'invalid image')
        expected_error = augment_data.cv2.error
    else:
        Path('yolo_dataset/labels/train/a.txt').write_text('3 0.3 0.5 0.1\n')
        expected_error = ValueError

    with pytest.raises(expected_error):
        augment_data.augment_dataset()

    assert snapshot(output) == before
