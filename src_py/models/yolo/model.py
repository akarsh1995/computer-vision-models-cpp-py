from torch.functional import Tensor
from torch.nn import BatchNorm2d, Conv2d, Linear, MaxPool2d
import torch.nn as nn
import torch


if __name__ == "__main__":
    input = torch.randn((1, 3, 448, 448))
    conv = Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    x = conv(input)
    print(x.shape)
    maxpool = MaxPool2d(kernel_size=2, stride=2)
    x = maxpool(x)
    print(x.shape)
    conv = Conv2d(64, 192, kernel_size=3, padding=1)
    x = conv(x)
    print(x.shape)
    maxpool = MaxPool2d(kernel_size=2, stride=2)  # stride=2, padding=1
    x = maxpool(x)
    print(x.shape)

    conv = Conv2d(192, 128, kernel_size=1)
    x = conv(x)
    conv = Conv2d(128, 256, kernel_size=3, padding=1)
    x = conv(x)
    conv = Conv2d(256, 256, kernel_size=1)
    x = conv(x)
    conv = Conv2d(256, 512, kernel_size=3, padding=1)
    x = conv(x)
    maxpool = MaxPool2d(kernel_size=2, stride=2)  # stride=2, padding=1
    x = maxpool(x)
    print(x.shape)

    conv = Conv2d(512, 256, kernel_size=1)
    x = conv(x)
    conv = Conv2d(256, 512, kernel_size=3, padding=1)
    x = conv(x)

    conv = Conv2d(512, 256, kernel_size=1)
    x = conv(x)
    conv = Conv2d(256, 512, kernel_size=3, padding=1)
    x = conv(x)

    conv = Conv2d(512, 256, kernel_size=1)
    x = conv(x)
    conv = Conv2d(256, 512, kernel_size=3, padding=1)
    x = conv(x)

    conv = Conv2d(512, 256, kernel_size=1)
    x = conv(x)
    conv = Conv2d(256, 512, kernel_size=3, padding=1)
    x = conv(x)

    conv = Conv2d(512, 512, kernel_size=1)
    x = conv(x)
    conv = Conv2d(512, 1024, kernel_size=3, padding=1)
    x = conv(x)
    maxpool = MaxPool2d(kernel_size=2, stride=2)  # stride=2, padding=1
    x = maxpool(x)
    print(x.shape)

    conv = Conv2d(1024, 512, kernel_size=1)
    x = conv(x)
    conv = Conv2d(512, 1024, kernel_size=3, padding=1)
    x = conv(x)

    conv = Conv2d(1024, 512, kernel_size=1)
    x = conv(x)
    conv = Conv2d(512, 1024, kernel_size=3, padding=1)
    x = conv(x)

    conv = Conv2d(1024, 1024, kernel_size=3, padding=1)
    x = conv(x)

    maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)  # stride=2, padding=1
    x = maxpool(x)

    print(x.shape)

    conv = Conv2d(1024, 1024, kernel_size=3, padding=1)
    x = conv(x)

    conv = Conv2d(1024, 1024, kernel_size=3, padding=1)
    x = conv(x)

    print(x.shape)

    flat = torch.nn.Flatten()

    x = flat(x)
    print(x.shape)

    fc = nn.Linear(7 * 7 * 1024, 4096)
    x = fc(x)
    fc = nn.Linear(4096, 7 * 7 * 30)
    x = fc(x)

    x = x.reshape((30, 7, 7))

    print(x.shape)
