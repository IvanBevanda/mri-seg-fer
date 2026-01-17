import sys
from argparse import ArgumentParser
import models
import importlib

SELECTED_EXP = None
PARAMS_FILE = None
LOGS_FILE = None
TESTX = None # .npy file containing 1-channel MRI images
TESTY = None # .npy file containing 1-channel segmentation masks
SELECTED_MODEL = None
LEARNING_RATE = None
OUTPUT_DIR = None

# TODO: Read from losses_pytorch/dice_loss.py, contains the loss functions. Calculate on entire test set (TESTY npy file)
def get_dice_coeff():
    pass

# TODO: Read from losses_pytorch/dice_loss.py, contains the loss functions. Calculate on entire test set (TESTY npy file)
def get_IoU():
    pass


# TODO: Read from log file
def get_losses():
    return None, None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--testx", type=str)
    parser.add_argument("--testy", type=str)
    parser.add_argument("--outputdir", type=str)
    args = parser.parse_args()
    SELECTED_EXP = args.exp
    TESTX = args.testx
    TESTY = args.testy
    OUTPUT_DIR = args.outputdir

    LOGS_FILE = SELECTED_EXP + "training.log"
    PARAMS_FILE = SELECTED_EXP + "params.pth"

    SELECTED_MODEL = None # TODO: Read from the log file
    MODEL_CONFIG = None # TODO: Read from the log file
    LEARNING_RATE = None # TODO: Read from the log file

    train_losses, test_losses = get_losses()
    # TODO: Plot training and testing losses on a single matplotlib plot, per epoch
    # TODO: Save the plot to OUTPUT_DIR/losses.png

    model = getattr(models, SELECTED_MODEL)().float().to(DEVICE)
    model.load_state_dict(torch.load(PARAMS_FILE))

    # TODO: Plot 3 images side by side: MRI image, label, model output
    # TODO: Save the plot to OUTPUT_DIR/example{n}.png
    # TODO: Do this for n examples

    dice_coeff = get_dice_coeff()
    iou = get_IoU()
    # TODO: print to stdout
    
