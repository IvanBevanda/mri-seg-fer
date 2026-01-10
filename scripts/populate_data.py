import numpy as np

TRAIN_SIZE = 100
TEST_SIZE = 10
IMG_WIDTH = 50
IMG_HEIGHT = 50
IMG_CHANNELS = (
    1  # DO NOT TOUCH TODO: Make it so generating_function works with other values too
)
"""
CrossEntropyLoss expects the (B, 1, H, W) format, where each pixel is a Long value corresponding to the class
"""
OUTPUT_DIR = "data/"


def generating_function(X: np.ndarray):
    return (X > 0.5).astype(np.long)


def create_X(n):
    return np.random.random(size=(n, IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT))


if __name__ == "__main__":
    train_X = create_X(TRAIN_SIZE)
    np.save(OUTPUT_DIR + "train_X.npy", train_X)
    np.save(OUTPUT_DIR + "train_Y.npy", generating_function(train_X))
    test_X = create_X(TEST_SIZE)
    np.save(OUTPUT_DIR + "test_X.npy", test_X)
    np.save(OUTPUT_DIR + "test_Y.npy", generating_function(test_X))
