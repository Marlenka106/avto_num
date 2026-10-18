"""Подготовка данных в формате YOLO."""
import shutil
from pathlib import Path

def prepare_yolo_dataset():
    # Исходные папки
    src_train = Path("number_plates/train")
    src_val = Path("number_plates/val")
    src_test = Path("number_plates/test")
    
    # Целевая структура
    dst_root = Path("yolo_dataset")
    dst_root.mkdir(exist_ok=True)
    
    for split in ["train", "val", "test"]:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Функция копирования файлов
    def copy_split(src_dir, split_name):
        count = 0
        for jpeg in src_dir.glob("*.jpeg"):
            stem = jpeg.stem
            txt = src_dir / f"{stem}.txt"
            
            if not txt.exists():
                print(f"⚠️  Пропущен {stem}: нет .txt файла")
                continue
            
            # Копируем изображение
            shutil.copy(jpeg, dst_root / "images" / split_name / jpeg.name)
            # Копируем аннотацию
            shutil.copy(txt, dst_root / "labels" / split_name / txt.name)
            count += 1
        
        print(f"✅ {split_name}: скопировано {count} пар (image + label)")
    
    # Копируем все сплиты
    copy_split(src_train, "train")
    copy_split(src_val, "val")
    copy_split(src_test, "test")
    
    print(f"\n🎉 Датасет подготовлен в {dst_root}")

if __name__ == "__main__":
    prepare_yolo_dataset()