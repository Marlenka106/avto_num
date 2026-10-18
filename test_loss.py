"""Тестирование функции потерь FCOS."""
import torch
from src.models.fcos import FCOSDetector
from src.models.loss import FCOSLoss

# Создаём модель и loss
print("Создание модели и функции потерь...")
model = FCOSDetector(num_classes=27, backbone_pretrained=True)
loss_fn = FCOSLoss(num_classes=27)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_fn = loss_fn.to(device)
print(f"✅ Модель и loss на устройстве: {device}")

# Создаём dummy input
dummy_input = torch.randn(2, 3, 512, 512).to(device)

# Создаём dummy targets
targets = [
    {
        'boxes': torch.tensor([[100, 100, 200, 200], [300, 150, 400, 250]], dtype=torch.float32).to(device),
        'labels': torch.tensor([5, 12], dtype=torch.long).to(device),
    },
    {
        'boxes': torch.tensor([[50, 50, 150, 150]], dtype=torch.float32).to(device),
        'labels': torch.tensor([3], dtype=torch.long).to(device),
    }
]

# Forward pass
print("\nЗапуск forward pass...")
model.train()
logits, bbox_reg, centerness = model(dummy_input)

# Вычисляем loss
print("Вычисление функции потерь...")
losses = loss_fn(logits, bbox_reg, centerness, targets)

print(f"\n✅ Loss вычислен успешно!")
print(f"Total loss: {losses['total_loss'].item():.4f}")
print(f"Classification loss: {losses['loss_cls'].item():.4f}")
print(f"Bounding box loss: {losses['loss_bbox'].item():.4f}")
print(f"Centerness loss: {losses['loss_centerness'].item():.4f}")

# Backward pass
print("\nПроверка backward pass...")
losses['total_loss'].backward()
print("✅ Backward pass успешен!")

print("\n🎉 Функция потерь полностью работоспособна!")