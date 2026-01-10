import torch.nn as nn


# Just a placeholder
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.placeholder = nn.Sequential(
            nn.Conv2d(1, 2, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(2, 2, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(2, 2, (3, 3), padding="same"),
        )

    def forward(self, x):
        return self.placeholder(x)
