import models
import torch.optim as optim
import torch.nn as nn
import importlib
import torch
import numpy as np
import tqdm
import logging

# Training config
NUM_EPOCHS = 2
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
LOGFILE = EXPPATH + "training.log"
PARAMFILE = EXPPATH + "params.pth"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def dynamic_import_loss_function(selected_loss_function):
    lf_module_name, lf_class_name = selected_loss_function.split(":")
    lf_module = importlib.import_module(lf_module_name)
    return getattr(lf_module, lf_class_name)


def train_epoch(model, optimizer, loss_function, trainx_path, trainy_path, batch_size):
    X = torch.from_numpy(np.load(trainx_path)).float()
    Y = torch.from_numpy(np.load(trainy_path)).long()
    dataset = torch.utils.data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )

    model.train()

    total_loss = 0
    for x, y in tqdm.tqdm(train_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        output = model(x)
        loss = loss_function(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def eval_model(model, loss_function, testx_path, testy_path, batch_size):
    X = torch.from_numpy(np.load(testx_path)).float()
    Y = torch.from_numpy(np.load(testy_path)).long()
    dataset = torch.utils.data.TensorDataset(X, Y)
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )

    model.eval()

    total_loss = 0
    for x, y in tqdm.tqdm(test_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        output = model(x)
        loss = loss_function(output, y)

        total_loss += loss.item()

    return total_loss


def train_network(
    selected_model,
    optim_config,
    selected_loss_function,
    lf_config,
    batch_size,
    trainx_path,
    trainy_path,
    testx_path,
    testy_path,
    num_epochs,
    logger,
    paramfile,
):

    model = getattr(models, selected_model)(*MODEL_CONFIG).float().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), **optim_config
    )  # TODO: Move optimizer selection into configs above
    loss_function = dynamic_import_loss_function(selected_loss_function)(*lf_config).to(
        DEVICE
    )

    min_test_loss = np.inf

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, optimizer, loss_function, trainx_path, trainy_path, batch_size
        )
        test_loss = eval_model(model, loss_function, testx_path, testy_path, 1)
        logger.info(
            f"\nEPOCH {epoch}:\nTRAINING_LOSS: {train_loss}\nTESTING_LOSS: {test_loss}"
        )
        print(f"Epoch {epoch} training loss: {train_loss}")
        print(f"Epoch {epoch} testing loss: {test_loss}")

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(model.state_dict(), paramfile)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=LOGFILE, level=logging.INFO)
    logger.info(
        f"""\nStarted training run with hyperparameters:
BATCH_SIZE: {BATCH_SIZE}
SELECTED_MODEL: {SELECTED_MODEL}
OPTIMIZER: Adam
LEARNING_RATE: {OPTIM_CONFIG['lr']}
LOSS_FUNCTION: {SELECTED_LOSS_FUNCTION}
LOSS_FUNCTION_CONFIG: {LF_CONFIG}
"""
    )

    train_network(
        SELECTED_MODEL,
        OPTIM_CONFIG,
        SELECTED_LOSS_FUNCTION,
        LF_CONFIG,
        BATCH_SIZE,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        NUM_EPOCHS,
        logger,
        PARAMFILE,
    )
