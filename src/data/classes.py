"""Классы символов номерных знаков."""

# Порядок совпадает с class_id в .txt аннотациях
# fmt: off
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'D', 'E', 'G', 'H', 'J', 'K', 'L', 'N',
           'R', 'S', 'T', 'U', 'V', 'X', 'Z']
# fmt: on


def read_plate(txt_path):
    """Строка номера из YOLO-разметки: символы слева направо."""
    boxes = [line.split() for line in txt_path.read_text().splitlines()]
    return "".join(CLASSES[int(box[0])] for box in sorted(boxes, key=lambda box: float(box[1])))
