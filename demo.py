"""Демонстрация: распознаёт номера на изображениях (папка или отдельные файлы).

Если рядом с изображением лежит .txt разметка, номер сравнивается с ней.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from evaluate import load_fcos, make_fcos_predictor, make_yolo_predictor
from src.data.classes import CLASSES, read_plate
from src.data.images import list_images
from src.evaluation.postprocess import filter_plate_predictions


def collect_images(paths):
    """Изображения из переданных файлов и папок."""
    images = []
    for path in paths:
        images.extend(list_images(path) if path.is_dir() else [path])
    return images


def recognize(images, predict):
    """Печатает распознанный номер для каждого изображения, возвращает число верных среди размеченных."""
    correct, labeled = 0, 0
    for img_path in images:
        pred = "".join(CLASSES[class_id] for class_id in filter_plate_predictions(predict(img_path)))
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"   {img_path.name:<32} {pred}")
            continue

        gt = read_plate(txt_path)
        correct += gt == pred
        labeled += 1
        status = "✅" if gt == pred else "❌"
        print(f"{status} {img_path.name:<32} {pred:<8} | GT: {gt}")
    return correct, labeled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, nargs="+", help="изображения или папки с изображениями")
    parser.add_argument("--model", choices=["yolo", "fcos"], default="yolo")
    parser.add_argument("--weights", required=True, help="путь к best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.model == "yolo":
        predict = make_yolo_predictor(YOLO(args.weights), args.conf)
    else:
        predict = make_fcos_predictor(load_fcos(args.weights), args.conf)

    correct, labeled = recognize(collect_images(args.images), predict)
    # Неполная разметка не исключается: на таких изображениях «❌» может быть ошибкой разметки
    if labeled:
        print(f"\nСовпало с разметкой: {correct}/{labeled}")


if __name__ == "__main__":
    main()
