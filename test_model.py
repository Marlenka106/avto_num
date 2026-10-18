"""Тестирование архитектуры FCOS модели."""
import torch
from src.models.fcos import FCOSDetector

# Создаём модель
print("Создание FCOS модели...")
model = FCOSDetector(num_classes=27, backbone_pretrained=True)
print(f"✅ Модель создана. Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

# Переводим на GPU если доступен
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"✅ Модель на устройстве: {device}")

# Создаём dummy input (батч из 2 изображений размером 3x512x512)
dummy_input = torch.randn(2, 3, 512, 512).to(device)

# Forward pass (БЕЗ torch.no_grad, чтобы сохранить градиенты)
print("\nЗапуск forward pass...")
model.train()
logits, bbox_reg, centerness = model(dummy_input)

print(f"✅ Forward pass успешен!")
print(f"Количество уровней пирамиды: {len(logits)}")
for i, (log, reg, cent) in enumerate(zip(logits, bbox_reg, centerness)):
    print(f"  Уровень {i}:")
    print(f"    Logits shape: {log.shape}")
    print(f"    BBox reg shape: {reg.shape}")
    print(f"    Centerness shape: {cent.shape}")

# Тест backward pass (градиенты)
print("\nПроверка backward pass...")
dummy_loss = sum(log.mean() + reg.mean() + cent.mean() 
                 for log, reg, cent in zip(logits, bbox_reg, centerness))
dummy_loss.backward()
print(f"✅ Backward pass успешен! Loss: {dummy_loss.item():.4f}")

# Проверяем, что градиенты действительно вычислились
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
               for p in model.parameters())
print(f"✅ Градиенты вычислены: {has_grad}")

print("\n🎉 Модель полностью работоспособна!")