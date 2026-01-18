import torch
import torch.nn as nn
import numpy as np


class UNet(nn.Module):

    def __init__(
        self, in_channels=1, out_channels=2, base_filters=16, apply_sigmoid=False
    ):
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
        self.c6 = conv2d_block(f * 16, f * 8)  # concat: (f*8 from skip) + (f*8 up)

        self.u7 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.c7 = conv2d_block(f * 8, f * 4)

        self.u8 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.c8 = conv2d_block(f * 4, f * 2)

        self.u9 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.c9 = conv2d_block(f * 2, f)

        # Output
        self.outputs = nn.Sequential(nn.Conv2d(f, out_channels, kernel_size=1))

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


# TODO: Refactor so it's actually maintainable
class UNetPlusPlus(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, base_filters=16):
        super(UNetPlusPlus, self).__init__()

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

        self.us = nn.Upsample(
            scale_factor=2
        )  # https://distill.pub/2016/deconv-checkerboard/

        # Encoder backbone
        self.x0_0 = conv2d_block(in_channels, f)
        self.p0_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.x1_0 = conv2d_block(f, f * 2)
        self.p1_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.x2_0 = conv2d_block(f * 2, f * 4)
        self.p2_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.x3_0 = conv2d_block(f * 4, f * 8)
        self.p3_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.x4_0 = conv2d_block(f * 8, f * 16)

        # Pyramidal skip layers (see original paper)
        # Horribly convoluted (pun intended)
        self.x0_1 = conv2d_block(f + 2 * f, f)
        self.x0_2 = conv2d_block(f + f + 2 * f, f)
        self.x0_3 = conv2d_block(f + f + f + 2 * f, f)

        self.x1_1 = conv2d_block(2 * f + 4 * f, 2 * f)
        self.x1_2 = conv2d_block(2 * f + 2 * f + 4 * f, 2 * f)

        self.x2_1 = conv2d_block(4 * f + 8 * f, 4 * f)

        # Decoder backbone
        self.x0_4 = conv2d_block(f + f + f + f + 2 * f, f)

        self.x1_3 = conv2d_block(2 * f + 2 * f + 2 * f + 4 * f, 2 * f)

        self.x2_2 = conv2d_block(4 * f + 4 * f + 8 * f, 4 * f)

        self.x3_1 = conv2d_block(8 * f + 16 * f, 8 * f)

        self.output = self.outputs = nn.Sequential(
            nn.Conv2d(f, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Encoder backbone propagation
        x0_0 = self.x0_0(x)
        p0_0 = self.p0_0(x0_0)
        x1_0 = self.x1_0(p0_0)
        p1_0 = self.p1_0(x1_0)
        x2_0 = self.x2_0(p1_0)
        p2_0 = self.p2_0(x2_0)
        x3_0 = self.x3_0(p2_0)
        p3_0 = self.p3_0(x3_0)
        x4_0 = self.x4_0(p3_0)

        # Layer 2 skip pathway propagation
        x2_1 = self.x2_1(torch.cat([self.us(x3_0), x2_0], dim=1))

        # Layer 1 skip pathway propagation
        x1_1 = self.x1_1(torch.cat([x1_0, self.us(x2_0)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self.us(x2_1)], dim=1))

        # Layer 0 skip pathway propagation
        x0_1 = self.x0_1(torch.cat([x0_0, self.us(x1_0)], dim=1))
        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self.us(x1_1)], dim=1))
        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self.us(x1_2)], dim=1))

        # Decoder backbone propagation
        x3_1 = self.x3_1(torch.cat([x3_0, self.us(x4_0)], dim=1))
        x2_2 = self.x2_2(torch.cat([x2_0, x2_1, self.us(x3_1)], dim=1))
        x1_3 = self.x1_3(torch.cat([x1_0, x1_1, x1_2, self.us(x2_2)], dim=1))
        x0_4 = self.x0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.us(x1_3)], dim=1))

        return self.output(x0_4)


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

if __name__ == "__main__":
    input_img = torch.Tensor(np.load("../data/test_X.npy"))
    model = UNet()
    print(input_img.shape)
    print(model(input_img).shape)
