import os
import numpy as np
import random
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

train_X = np.load(os.path.join(OUTPUT_DIR, "train_X.npy"))
train_Y = np.load(os.path.join(OUTPUT_DIR, "train_Y.npy"))
test_X = np.load(os.path.join(OUTPUT_DIR, "test_X.npy"))
test_Y = np.load(os.path.join(OUTPUT_DIR, "test_Y.npy"))

print("Loaded files from output folder:")
print("train_X:", train_X.shape)
print("train_Y:", train_Y.shape)
print("test_X :", test_X.shape)
print("test_Y :", test_Y.shape)


def check_dataset(X, Y, name="dataset"):
    assert X.shape[0] == Y.shape[0], f"{name}: number of examples doesn't match!"

    problems = 0
    for i in range(len(X)):
        if X[i].shape != Y[i].shape:
            print(f"{name}: Shape mismatch on index {i} -> X: {X[i].shape}, Y: {Y[i].shape}")
            problems += 1
        elif X[i].max() > 1.0 or X[i].min() < 0.0:
            print(f"{name}: Pixel values outside [0,1] on index {i}")
            problems += 1
        elif not np.any(Y[i]):
            print(f"{name}: Mask empty on index {i}")

    if problems == 0:
        print(f"{name}: All examples are correctly paired!")
    else:
        print(f"{name}: found {problems} problems in dataset!")


check_dataset(train_X, train_Y, "Train set")
check_dataset(test_X, test_Y, "Test set")

r = random.randint(0, len(train_X) - 1)
print(f"\nExample on index {r}")
print("Image shape :", train_X[r].shape)
print("Mask shape :", train_Y[r].shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(train_X[r].squeeze(), cmap='gray')
ax[0].set_title("MRI Image")
ax[1].imshow(train_Y[r].squeeze(), cmap='gray')
ax[1].set_title("Tumor mask")
plt.show()
