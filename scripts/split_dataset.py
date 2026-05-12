import os
import shutil
import random
from glob import glob

# --- CONFIG ---
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")
SPLIT = [0.75, 0.15, 0.10]  # train, val, test

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and d not in ["train", "val", "test"]]

for cls in classes:
    imgs = glob(os.path.join(DATASET_DIR, cls, "*.jpg"))
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])
    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train+n_val]
    test_imgs = imgs[n_train+n_val:]

    for target_dir, img_list in zip([TRAIN_DIR, VAL_DIR, TEST_DIR], [train_imgs, val_imgs, test_imgs]):
        cls_dir = os.path.join(target_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for img in img_list:
            shutil.copy(img, cls_dir)

print("✅ Dataset split into train/val/test.")
