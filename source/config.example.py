# Training config
NUM_EPOCHS = 10
BATCH_SIZE = 8
OPTIM_CONFIG = {"lr": 1e-4}
SELECTED_MODEL = "UNet"
MODEL_CONFIG = {}
SELECTED_LOSS_FUNCTION = "losses_pytorch.dice_loss:DC_and_CE_loss"  # https://github.com/JunMa11/SegLossOdyssey/tree/master
LF_CONFIG = [{"batch_dice": True, "smooth": 1e-5, "do_bg": False, "square": False}, {}]
X_TRAIN_PATH = "output/train_X.npy"
Y_TRAIN_PATH = "output/train_Y.npy"
X_TEST_PATH = "output/test_X.npy"
Y_TEST_PATH = "output/test_Y.npy"
# Change manually! Create .../expN directory manually!
EXPPATH = "output/experiments/exp1/"
