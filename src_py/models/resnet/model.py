from typing import List
import torch
import torch.nn as nn
from torch.nn.functional import relu


class CommonBlock(nn.Module):
    def __init__(self, output_channel, block_channel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential()
        self.block.extend(
            [
                nn.Conv2d(output_channel, block_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(block_channel),
                nn.ReLU(),
                # padding of 1 is needed to maintain the channel width and height shape
                nn.Conv2d(
                    block_channel, block_channel, kernel_size=3, stride=1, padding=1
                ),
                nn.BatchNorm2d(block_channel),
                nn.ReLU(),
                nn.Conv2d(block_channel, output_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(output_channel),
            ]
        )

    def forward(self, x):
        return relu(self.block(x) + x)


class BlockGroup(nn.Module):
    def __init__(
        self,
        input_channel: int,
        block_channel: int,
        output_channel: int,
        first_layer_center_conv_stride: int,
        repetitions: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.downsampling_layer = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=3,
            stride=first_layer_center_conv_stride,
            padding=1,
        )
        self.first_block = nn.Sequential(
            *[
                nn.Conv2d(input_channel, block_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(block_channel),
                nn.ReLU(),
                nn.Conv2d(
                    block_channel,
                    block_channel,
                    kernel_size=3,
                    # in the first block repetition stride of 1
                    # because the channel dim is halved by maxpooling in the beginning.
                    # whereas in rest of the block repetitions stride of 2 is required to
                    # half the width and height of the channel
                    stride=first_layer_center_conv_stride,
                    padding=1,
                ),
                nn.BatchNorm2d(block_channel),
                nn.ReLU(),
                nn.Conv2d(block_channel, output_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(output_channel),
            ]
        )

        self.seq = nn.Sequential()
        for _ in range(1, repetitions):
            self.seq.append(CommonBlock(output_channel, block_channel))

    def forward(self, x):
        # for every first block, upsampling needs to be done to propogate values to the next block
        upsampled = self.downsampling_layer(x)
        x = torch.relu(self.first_block(x) + upsampled)
        return self.seq(x)


class ResNet50(nn.Module):
    def __init__(self, repetitions: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Not mentioned in the paper but padding of 3 and stride of 2 is required
        # output_width = (input_width - kernel_width + 2 * padding) / stride + 1
        # output_height = (input_height - kernel_height + 2 * padding) / stride + 1
        # ((224 - 7 + 2 * 3) / 2) + 1
        # 111 + 1 = 112 x 112
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        # ((112 - 3 + 2 * 1) / 2) + 1
        #  55 + 1 = 56 x 56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block_groups = nn.Sequential()

        first_layer_center_conv_stride = 1  # 1, 2, 2, 2
        input_channel = 64  # 64, 256, 512, 1024
        block_channel = 64  # 64, 128, 256, 512
        output_channel = block_channel * 4  # 256, 512, 1024, 2048

        for r in repetitions:
            self.block_groups.append(
                BlockGroup(
                    input_channel,
                    block_channel,
                    output_channel,
                    first_layer_center_conv_stride,
                    r,
                )
            )

            first_layer_center_conv_stride = 2
            input_channel = output_channel
            block_channel = block_channel * 2  # 64 * 2
            output_channel = block_channel * 4  # 64 * 2 * 4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(input_channel, 1000)

    def forward(self, x):
        x = self.maxpool(self.conv(x))
        x = self.block_groups(x)
        x = self.avgpool(x)
        x = self.fc(x.flatten(1))
        return nn.functional.softmax(x, dim=1)
