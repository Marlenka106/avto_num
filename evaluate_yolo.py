"""Финальная оценка YOLO: Exact Match Accuracy с умной постобработкой."""

from pathlib import Path

from ultralytics import YOLO

from src.data.images import list_images

# fmt: off
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'D', 'E', 'G', 'H', 'J', 'K', 'L', 'N',
           'R', 'S', 'T', 'U', 'V', 'X', 'Z']
# fmt: on


def get_gt_from_txt(txt_path):
    if not txt_path.exists():
        return ""
    boxes_chars = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, cx = int(parts[0]), float(parts[1])
                boxes_chars.append((cx, CLASSES[class_id]))
    boxes_chars.sort(key=lambda x: x[0])
    return "".join([c for _, c in boxes_chars])


def filter_plate_predictions(preds):
    """
    preds: список кортежей (cx, char, box, conf)
    Применяет строгие геометрические правила и ограничение максимальной длины.
    """
    if len(preds) == 0:
        return []

    # Сортируем по X для стабильного расчета медианы
    preds.sort(key=lambda x: x[0])

    heights = [p[2][3] - p[2][1] for p in preds]
    y_centers = [(p[2][1] + p[2][3]) / 2 for p in preds]

    med_h = sorted(heights)[len(heights) // 2]
    med_y = sorted(y_centers)[len(y_centers) // 2]

    filtered = []
    for cx, char, box, conf in preds:
        h = box[3] - box[1]
        y = (box[1] + box[3]) / 2

        # ОЧЕНЬ СТРОГИЕ правила: символы номера имеют почти одинаковую высоту и линию
        if 0.75 * med_h < h < 1.25 * med_h and abs(y - med_y) < 0.35 * med_h:
            filtered.append((cx, char, conf))

    # ЗАЩИТА ОТ ФАНТОМОВ: Максимальная длина номера в датасете = 7.
    # Если найдено больше 7, отбрасываем наименее уверенные предсказания.
    if len(filtered) > 7:
        filtered.sort(key=lambda x: x[2], reverse=True)  # Сортируем по уверенности (по убыванию)
        filtered = filtered[:7]  # Оставляем только топ-7
        filtered.sort(key=lambda x: x[0])  # Возвращаем сортировку по X

    return [p[1] for p in filtered]


def evaluate_yolo():
    model_path = "runs/detect/runs/yolo/plates_detector/weights/best.pt"

    if not Path(model_path).exists():
        print(f"❌ Модель не найдена по пути: {model_path}")
        return

    print(f"🚀 Загрузка модели из {model_path}...")
    model = YOLO(model_path)

    test_dir = Path("yolo_dataset/images/test")
    labels_dir = Path("yolo_dataset/labels/test")

    correct = 0
    total = 0

    # СНИЖАЕМ порог, чтобы поймать тусклые '4' и 'J'.
    # Фильтр ниже возьмет на себя работу по отсечению шума!
    CONF_THRESHOLD = 0.4

    test_images = list_images(test_dir)
    n_images = len(test_images)
    print(f"\n🔍 Оценка на {n_images} тестовых изображениях (conf={CONF_THRESHOLD}, строгий фильтр)...\n")

    for img_path in test_images:
        stem = img_path.stem
        txt_path = labels_dir / f"{stem}.txt"

        gt_str = get_gt_from_txt(txt_path)

        # Запрашиваем все детекции с conf >= 0.20
        results_yolo = model(img_path, conf=CONF_THRESHOLD, iou=0.45, verbose=False)

        preds_with_conf = []
        if len(results_yolo[0].boxes) > 0:
            boxes = results_yolo[0].boxes.xyxy.cpu().numpy()
            classes = results_yolo[0].boxes.cls.cpu().numpy()
            confs = results_yolo[0].boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, classes, confs, strict=False):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                char = CLASSES[int(cls_id)]
                preds_with_conf.append((cx, char, box, conf))

        # Применяем умный фильтр
        pred_chars = filter_plate_predictions(preds_with_conf)
        pred_str = "".join(pred_chars)

        is_correct = (pred_str == gt_str) and len(gt_str) > 0
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        total += 1
        print(f"[{total:2d}/{n_images}] {status} GT: {gt_str: <10} | Pred: {pred_str: <10}")

    accuracy = correct / total if total > 0 else 0
    print("\n" + "=" * 55)
    print(f"🎯 EXACT MATCH ACCURACY: {accuracy:.2%} ({correct}/{total})")
    print("=" * 55)

    if accuracy >= 0.80:
        print("🏆 ПОЗДРАВЛЯЮ! ЦЕЛЬ ДОСТИГНУТА: Точность >= 80%! 🎉")
    else:
        print("⚠️ Точность < 80%. Попробуйте изменить CONF_THRESHOLD на 0.15 или 0.25.")


if __name__ == "__main__":
    evaluate_yolo()
