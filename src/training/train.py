"""Полный цикл обучения FCOS детектора (СТАБИЛЬНАЯ ВЕРСИЯ БЕЗ NaN)."""

import json
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from src.data.dataset import PlateCharDataset, get_transform
from src.evaluation.inference import decode_predictions, to_detections
from src.evaluation.postprocess import filter_plate_predictions
from src.models.fcos import FCOSDetector, compute_locations
from src.models.loss import FCOSLoss
from src.utils.seed import set_seed


def collate_fn(batch):
    """Складывает изображения в один тензор (все уже 256x512)."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def seed_worker(worker_id):
    """Задаёт Albumentations отдельный воспроизводимый seed для каждого worker и эпохи."""
    worker = get_worker_info()
    worker.dataset.transforms.set_random_seed(torch.initial_seed() % (2**32))


def train_one_epoch(model, loss_fn, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Forward Pass (БЕЗ Mixed Precision для стабильности)
        logits, bbox_reg, centerness = model(images)
        losses = loss_fn(logits, bbox_reg, centerness, targets)
        loss = losses['total_loss']

        # Backward Pass
        loss.backward()

        # Gradient Clipping (защита от взрыва градиентов)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, loss_fn, dataloader, device):
    """Возвращает средний val loss и Exact Match - долю номеров, распознанных целиком."""
    model.eval()
    total_loss = 0
    correct, total = 0, 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        logits, bbox_reg, centerness = model(images)
        losses = loss_fn(logits, bbox_reg, centerness, targets)
        total_loss += losses['total_loss'].item()

        # Та же постобработка, что в evaluate.py; GT сортируем слева направо
        preds = decode_predictions(logits, bbox_reg, centerness, compute_locations(logits), plate_filter=False)
        for pred, target in zip(preds, targets, strict=True):
            gt_labels = target['labels'][target['boxes'][:, 0].argsort()].tolist()
            correct += filter_plate_predictions(to_detections(pred)) == gt_labels
            total += 1

    return total_loss / len(dataloader), correct / total


def main():
    # 1. Настройки
    out_dir = Path("weights")
    # Прошлый прогон не перезаписываем: при падении до первой эпохи оценивались бы старые веса
    if (out_dir / "best.pt").exists():
        raise SystemExit(f"{out_dir / 'best.pt'} уже есть - перенесите прошлый прогон")

    seed = 42
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    epochs = 60
    batch_size = 4
    lr = 1e-4  # Уменьшенный learning rate

    # 2. Данные (УКАЗЫВАЕМ НА АУГМЕНТИРОВАННЫЙ ДАТАСЕТ)
    logger.info("Loading augmented datasets...")
    train_dataset = PlateCharDataset(
        img_dir="yolo_dataset_augmented/images/train",
        txt_dir="yolo_dataset_augmented/labels/train",
        transforms=get_transform(train=True, seed=seed),
    )
    val_dataset = PlateCharDataset(
        img_dir="yolo_dataset_augmented/images/val",
        txt_dir="yolo_dataset_augmented/labels/val",
        transforms=get_transform(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # 3. Модель и оптимизатор
    logger.info("Initializing model...")
    model = FCOSDetector(num_classes=27, backbone_pretrained=True).to(device)
    loss_fn = FCOSLoss(num_classes=27).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + Cosine Annealing
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - 5) / (epochs - 5) * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. Цикл обучения
    best_val_acc = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_exact_match': []}

    logger.info("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, loss_fn, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_exact_match'].append(val_acc)

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Exact Match: {val_acc:.2%} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Лучшая модель - по целевой метрике, а не по loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "best.pt")
            logger.info(f"💾 Saved new best model with val exact match: {best_val_acc:.2%}")

        # История пишется каждую эпоху - при падении обучения она не теряется
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    logger.success("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()
