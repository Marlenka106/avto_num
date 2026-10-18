# 🚗 License Plate Character Detection & OCR

Детектор символов номерных знаков с использованием кастомной архитектуры FCOS и YOLOv8.

## 📊 Результаты

### YOLOv8 (Production-ready)
- **mAP50: 96.9%** (валидация)
- **Exact Match Accuracy: 75%** (тестовая выборка, 20 изображений)
- **Скорость инференса: ~10ms** на RTX 4060

### FCOS (Кастомная архитектура)
- mAP50: ~60% (недообучена из-за малого датасета)
- Full pipeline: backbone → FPN → FCOS head → NMS

## 🏗️ Архитектура проекта
