import os
import numpy as np
import cv2
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "..", "data", "archive", "brisc2025", "segmentation_task"
)

DATA_DIR = os.path.abspath(DATA_DIR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Output folder:", OUTPUT_DIR)

IMG_SIZE  = (256, 256)
CHANNELS  = 1

def load_images_and_masks(split):
    images_dir = os.path.join(DATA_DIR, split, "images")
    masks_dir  = os.path.join(DATA_DIR, split, "masks")

    X, Y = [], []

    for fname in tqdm(os.listdir(images_dir)):
        img_path  = os.path.join(images_dir, fname)
        mask_name = fname.replace(".jpg", ".png")
        mask_path = os.path.join(masks_dir, mask_name)

        if not os.path.exists(mask_path):
            print("Missing mask for:", fname)
            continue

        img  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img  = cv2.resize(img,  IMG_SIZE)
        mask = cv2.resize(mask, IMG_SIZE)

        img  = img.astype(np.float32) / 255.0

        mask  = (mask > 0).astype(np.uint8)

        img   = np.expand_dims(img,  axis=0)
        mask  = np.expand_dims(mask, axis=0)

        X.append(img)
        Y.append(mask)

    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y

train_X, train_Y = load_images_and_masks("train")
np.save(os.path.join(OUTPUT_DIR, "train_X.npy"), train_X)
np.save(os.path.join(OUTPUT_DIR, "train_Y.npy"), train_Y)


test_X, test_Y = load_images_and_masks("test")
np.save(os.path.join(OUTPUT_DIR, "test_X.npy"), test_X)
np.save(os.path.join(OUTPUT_DIR, "test_Y.npy"), test_Y)

print("train_X:",  train_X.shape)
print("train_Y:",  train_Y.shape)
print("test_X:",   test_X.shape)
print("test_Y:",   test_Y.shape)
