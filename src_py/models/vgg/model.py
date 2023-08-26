from typing import List, Union
import torch
import torch.nn as nn


VGG16 = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]


class VGGNet(nn.Module):
    in_channels: int

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        im_shape: tuple[int, int],
        vgg_map: List[Union[int, str]] = VGG16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.im_shape = im_shape
        self.conv_layers = self.create_conv_layers(vgg_map)
        input_features_after_convs = (
            self.conv_layers(torch.rand(1, in_channels, im_shape[0], im_shape[1]))
            .flatten(1)
            .shape[1]
        )
        self.fcs = nn.Sequential(
            *[
                nn.Linear(input_features_after_convs, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, self.num_classes),
            ]
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fcs(x.flatten(1))

    def create_conv_layers(self, architecture):
        layers: list[nn.modules.Module] = []
        in_channels = self.in_channels

        for layer in architecture:
            if isinstance(layer, int):
                out_channels = layer
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        nn.ReLU(),
                    ]
                )
                in_channels = out_channels
            elif isinstance(layer, str) and layer == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def gen_random_input(self):
        return torch.rand((1, self.in_channels, self.im_shape[0], self.im_shape[1]))
