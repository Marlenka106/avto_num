"""Классы символов номерных знаков и чтение разметки."""

# Порядок совпадает с class_id в .txt аннотациях
# fmt: off
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'D', 'E', 'G', 'H', 'J', 'K', 'L', 'N',
           'R', 'S', 'T', 'U', 'V', 'X', 'Z']
# fmt: on

# Ошибки разметки: номер на изображении отличается от разметки одним символом (проверено визуально).
# Без исправления кадр того же автомобиля попадает в другую группу сплита - утечка между train и val
LABEL_FIXES = {
    'ckn75td6o000l3068nelfuzlo': '1000LHA',  # размечен 0000LHA
    'ckie5iepj004p246a3vsgr3xv': '3100NKD',  # размечен 3000NKD
    'ckn7dpc87001w3h66nfp0otgl': '4949UVB',  # размечен 4949UDB
    'ckicyspah001d246alfgflkl1': '9454BKJ',  # размечен 8454BKJ
}


def read_label(txt_path):
    """Строки YOLO-разметки с исправленными классами из LABEL_FIXES."""
    lines = txt_path.read_text().splitlines()
    plate = LABEL_FIXES.get(txt_path.stem)
    if plate is None:
        return lines

    # Символы номера идут слева направо, строки в .txt - в произвольном порядке
    order = sorted(range(len(lines)), key=lambda i: float(lines[i].split()[1]))
    for i, char in zip(order, plate, strict=True):
        lines[i] = " ".join([str(CLASSES.index(char)), *lines[i].split()[1:]])
    return lines


def read_plate(txt_path):
    """Строка номера из YOLO-разметки: символы слева направо."""
    boxes = [line.split() for line in read_label(txt_path)]
    return "".join(CLASSES[int(box[0])] for box in sorted(boxes, key=lambda box: float(box[1])))
