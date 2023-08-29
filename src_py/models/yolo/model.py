from torch.functional import Tensor
from torch.nn import BatchNorm2d, Conv2d, Flatten, MaxPool2d
import torch.nn as nn
import torch


class Conv2dBNReLU(nn.Module):
    def __init__(self, *args, leaky=True, **kwargs) -> None:
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            Conv2d(*args, **kwargs),
            BatchNorm2d(args[1]),
            nn.LeakyReLU(0.1) if leaky else nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class Yolo(nn.Module):
    def __init__(
        self,
        n_cells_per_row: int,
        n_bbox_predictors: int,
        n_classes: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.n_bbox_predictors = n_bbox_predictors
        self.n_cells_per_row = n_cells_per_row

        self.layers = nn.Sequential(
            Conv2dBNReLU(3, 64, kernel_size=7, stride=2, padding=3),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2dBNReLU(64, 192, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=2, stride=2),  # stride=2, padding=1
            Conv2dBNReLU(192, 128, kernel_size=1),
            Conv2dBNReLU(128, 256, kernel_size=3, padding=1),
            Conv2dBNReLU(256, 256, kernel_size=1),
            Conv2dBNReLU(256, 512, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=2, stride=2),  # stride=2, padding=1
            Conv2dBNReLU(512, 256, kernel_size=1),
            Conv2dBNReLU(256, 512, kernel_size=3, padding=1),
            Conv2dBNReLU(512, 256, kernel_size=1),
            Conv2dBNReLU(256, 512, kernel_size=3, padding=1),
            Conv2dBNReLU(512, 256, kernel_size=1),
            Conv2dBNReLU(256, 512, kernel_size=3, padding=1),
            Conv2dBNReLU(512, 256, kernel_size=1),
            Conv2dBNReLU(256, 512, kernel_size=3, padding=1),
            Conv2dBNReLU(512, 512, kernel_size=1),
            Conv2dBNReLU(512, 1024, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=2, stride=2),  # stride=2, padding=1,
            Conv2dBNReLU(1024, 512, kernel_size=1),
            Conv2dBNReLU(512, 1024, kernel_size=3, padding=1),
            Conv2dBNReLU(1024, 512, kernel_size=1),
            Conv2dBNReLU(512, 1024, kernel_size=3, padding=1),
            Conv2dBNReLU(1024, 1024, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=3, stride=2, padding=1),  # stride=2, padding=1,
            Conv2dBNReLU(1024, 1024, kernel_size=3, padding=1),
            Conv2dBNReLU(1024, 1024, kernel_size=3, padding=1, leaky=False),
            Flatten(),
            nn.Linear(n_cells_per_row * n_cells_per_row * 1024, 4096),
            nn.Linear(
                4096,
                n_cells_per_row * n_cells_per_row * (n_classes + 5 * n_bbox_predictors),
            ),
        )

    def forward(self, x):
        return self.layers(x).view(
            (
                -1,
                (self.n_classes + 5 * self.n_bbox_predictors),
                self.n_cells_per_row,
                self.n_cells_per_row,
            )
        )


if __name__ == "__main__":
    input = torch.randn((1, 3, 448, 448))
    y = Yolo(7, 2, 20)
    x = y(input)
    print(x.shape)
