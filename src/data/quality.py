"""Проверка качества разметки номеров."""

# Полный саудовский номер: 4 цифры + 3 буквы. Разметка другой длины в датасете -
# почти всегда частично размеченный номер или дубль бокса
PLATE_LENGTH = 7
# Символ занимает больше половины высоты кропа; боксы ниже 10% - ошибочные клики разметчика
MIN_BOX_HEIGHT = 0.1


def is_clean_annotation(txt_path):
    """True, если номер размечен полностью и без вырожденных боксов."""
    heights = [float(line.split()[4]) for line in txt_path.read_text().splitlines()]
    return len(heights) == PLATE_LENGTH and min(heights) >= MIN_BOX_HEIGHT
