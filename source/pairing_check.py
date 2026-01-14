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

print("Učitani fajlovi iz output foldera:")
print("train_X:", train_X.shape)
print("train_Y:", train_Y.shape)
print("test_X :", test_X.shape)
print("test_Y :", test_Y.shape)


def check_dataset(X, Y, name="dataset"):
    assert X.shape[0] == Y.shape[0], f"{name}: broj uzoraka se ne poklapa!"

    problems = 0
    for i in range(len(X)):
        if X[i].shape != Y[i].shape:
            print(f"{name}: Shape mismatch na indeksu {i} -> X: {X[i].shape}, Y: {Y[i].shape}")
            problems += 1
        elif X[i].max() > 1.0 or X[i].min() < 0.0:
            print(f"{name}: Pixel vrijednosti slike izvan [0,1] na indeksu {i}")
            problems += 1
        elif not np.any(Y[i]):
            print(f"{name}: Maska je prazna na indeksu {i}")

    if problems == 0:
        print(f"{name}: svi parovi su OK!")
    else:
        print(f"{name}: pronađeno {problems} problema u datasetu!")


check_dataset(train_X, train_Y, "Train set")
check_dataset(test_X, test_Y, "Test set")

r = random.randint(0, len(train_X) - 1)
print(f"\nPrikaz uzorka indeks {r}")
print("Shape slike :", train_X[r].shape)
print("Shape maske :", train_Y[r].shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(train_X[r].squeeze(), cmap='gray')
ax[0].set_title("MRI Slika")
ax[1].imshow(train_Y[r].squeeze(), cmap='gray')
ax[1].set_title("Maska Tumora")
plt.show()
