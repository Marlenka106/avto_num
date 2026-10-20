"""Метрики распознавания номеров."""

import math


def levenshtein(a, b):
    """Минимальное число вставок, удалений и замен символов, превращающее a в b."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def cer(gt, pred):
    """Character Error Rate: доля ошибочных символов относительно длины разметки."""
    return levenshtein(gt, pred) / len(gt)


def wilson_interval(successes, total, z=1.96):
    """95% доверительный интервал Уилсона для доли правильных ответов."""
    p = successes / total
    center = (p + z**2 / (2 * total)) / (1 + z**2 / total)
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    # Обрезаем погрешность float на краях, иначе при 0 правильных печатается «-0.0%»
    return max(0.0, center - half), min(1.0, center + half)
