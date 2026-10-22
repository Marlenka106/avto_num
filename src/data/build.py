"""Сборка датасета с сохранением предыдущего результата при ошибках."""

import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

from src.data.yolo_config import write_data_yaml


@contextmanager
def staged_dataset(destination):
    """Собирает датасет отдельно и заменяет целевую папку только после успеха."""
    destination = Path(destination)
    # Соседняя папка находится на той же файловой системе: замена делается rename.
    workspace = Path(mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    staged = workspace / 'new'
    backup = workspace / 'previous'
    published = False
    try:
        staged.mkdir()
        yield staged
        write_data_yaml(staged, dataset_root=destination)

        try:
            if destination.exists():
                destination.rename(backup)
            staged.rename(destination)
        except BaseException:
            if backup.exists():
                try:
                    backup.rename(destination)
                except OSError as error:
                    raise OSError(
                        f'Не удалось восстановить {destination}; предыдущий датасет сохранён в {backup}'
                    ) from error
            raise
        published = True
    finally:
        # Если откат тоже не удался, сохраняем резервную копию для восстановления.
        if published or not backup.exists():
            shutil.rmtree(workspace, ignore_errors=True)
