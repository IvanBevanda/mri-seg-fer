import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import models
from losses_pytorch.dice_loss import GDiceLossV2, IoULoss
from tqdm import tqdm
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_losses(log_file):
    train_losses, test_losses = [], []
    model_name, lr = "UNet", 0.001
    try:
        with open(log_file, "r") as f:
            for line in f:
                if "SELECTED_MODEL:" in line:
                    model_name = line.split(":")[-1].strip()
                if "LEARNING_RATE:" in line:
                    lr = float(line.split(":")[-1].strip())
                if "TRAINING_LOSS:" in line:
                    train_losses.append(float(line.split(":")[-1].strip()))
                if "TESTING_LOSS:" in line:
                    test_losses.append(float(line.split(":")[-1].strip()))
    except FileNotFoundError:
        print("Error!")
    return train_losses, test_losses, model_name, lr


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--testx", type=str, required=True)
    parser.add_argument("--testy", type=str, required=True)
    parser.add_argument("--outputdir", type=str, default="results")
    args = parser.parse_args()

    LOGS_FILE = args.exp + "training.log"
    PARAMS_FILE = args.exp + "params.pth"

    dice_criterion = GDiceLossV2(apply_nonlin=lambda t: torch.softmax(t, dim=1))
    iou_criterion = IoULoss(
        apply_nonlin=lambda t: torch.softmax(t, dim=1), do_bg=False, smooth=1e-5
    )

    train_l, test_l, SELECTED_MODEL, LEARNING_RATE = get_losses(LOGS_FILE)
    print(train_l, test_l)
    if train_l:
        plt.plot(train_l, label="Train")
        plt.plot(test_l, label="Test")
        plt.legend()
        plt.savefig(f"{args.outputdir}/losses.png")
        plt.close()

    model = getattr(models, SELECTED_MODEL)().float().to(DEVICE)
    model.load_state_dict(torch.load(PARAMS_FILE))
    model.eval()

    TESTX_arr = np.load(args.testx)
    TESTY_arr = np.load(args.testy)
    X_tensor = torch.from_numpy(TESTX_arr).float()
    Y_tensor = torch.from_numpy(TESTY_arr).float()

    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False
    )
    num = 0
    avg_dice = 0
    avg_iou = 0
    preds = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output = model(x)
            avg_dice += -dice_criterion(output, y)
            avg_iou += -iou_criterion(output, y)

            num += 1

            for_plot = np.argmax(output.cpu().detach().numpy(), axis=1).squeeze()
            preds.append(for_plot)
    avg_dice /= num
    avg_iou /= num  # Average might not be the best...
    print(f"Mean IoU (mean of IoU scores for each testing image): {avg_iou}")
    print(f"Mean Dice (mean of Dice scores for each testing image): {avg_dice}")

    for i in range(min(10, len(TESTX_arr))):
        fig, ax = plt.subplots(1, 3)
        random_index = random.randint(0, len(TESTX_arr) - 1)
        ax[0].imshow(TESTX_arr[random_index, 0], cmap="gray")
        ax[1].imshow(TESTY_arr[random_index, 0], cmap="gray")
        ax[2].imshow(preds[random_index], cmap="gray")
        plt.savefig(f"{args.outputdir}/example_{i}.png")
        plt.close()
    # python postprocessing.py --exp "../output/experiments/exp1/" --testx "../output/test_X.npy" --testy "../output/test_Y.npy" --outputdir "../output/TinoIPetra"
