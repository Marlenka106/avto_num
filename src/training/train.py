"""Полный цикл обучения FCOS детектора (СТАБИЛЬНАЯ ВЕРСИЯ БЕЗ NaN)."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from loguru import logger

from src.data.dataset import PlateCharDataset, get_transform
from src.models.fcos import FCOSDetector
from src.models.loss import FCOSLoss
from src.utils.seed import set_seed


def collate_fn(batch):
    """Складывает изображения в один тензор (все уже 256x512)."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


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
    model.eval()
    total_loss = 0
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        logits, bbox_reg, centerness = model(images)
        losses = loss_fn(logits, bbox_reg, centerness, targets)
        total_loss += losses['total_loss'].item()
    
    return total_loss / len(dataloader)


def main():
    # 1. Настройки
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    epochs = 60
    batch_size = 2
    lr = 1e-4  # Уменьшенный learning rate
    
    # 2. Данные
    logger.info("Loading datasets...")
    train_dataset = PlateCharDataset(
        img_dir="number_plates/train",
        txt_dir="number_plates/train",
        transforms=get_transform(train=True)
    )
    val_dataset = PlateCharDataset(
        img_dir="number_plates/val",
        txt_dir="number_plates/val",
        transforms=get_transform(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
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
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    logger.info("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device)
        val_loss = validate(model, loss_fn, val_loader, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path("weights/best.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"💾 Saved new best model with val_loss: {best_val_loss:.4f}")
    
    with open("weights/history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    logger.success("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()