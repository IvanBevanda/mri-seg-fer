import torch
import torch.nn as nn
import numpy as np

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, base_filters=16, apply_sigmoid=False):
        super(UNet, self).__init__()
        self.apply_sigmoid = apply_sigmoid

        def conv2d_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),

                nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        f = base_filters

        # Encoder
        self.c1 = conv2d_block(in_channels, f)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2 = conv2d_block(f, f * 2)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3 = conv2d_block(f * 2, f * 4)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c4 = conv2d_block(f * 4, f * 8)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.c5 = conv2d_block(f * 8, f * 16)

        # Decoder
        self.u6 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.c6 = conv2d_block(f * 16, f * 8)   # concat: (f*8 from skip) + (f*8 up)

        self.u7 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.c7 = conv2d_block(f * 8, f * 4)

        self.u8 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2, output_padding=1)
        self.c8 = conv2d_block(f * 4, f * 2)

        self.u9 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.c9 = conv2d_block(f * 2, f)

        # Output
        self.outputs = nn.Sequential(
            nn.Conv2d(f, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        # Bottleneck
        c5 = self.c5(p4)

        # Decoder + skip connections
        u6 = self.u6(c5)
        u6 = torch.cat([c4, u6], dim=1)
        c6 = self.c6(u6)

        u7 = self.u7(c6)
        u7 = torch.cat([c3, u7], dim=1)
        c7 = self.c7(u7)

        u8 = self.u8(c7)
        u8 = torch.cat([c2, u8], dim=1)
        c8 = self.c8(u8)

        u9 = self.u9(c8)
        u9 = torch.cat([c1, u9], dim=1)
        c9 = self.c9(u9)

        logits = self.outputs(c9)

        if self.apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


"""
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
"""

if __name__ == '__main__':
    input_img = torch.Tensor(np.load("../data/test_X.npy"))
    model = UNet()
    print(input_img.shape)
    print(model(input_img).shape)