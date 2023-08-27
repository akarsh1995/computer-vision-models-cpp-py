import torch
import torch.nn as nn
from torch.nn.functional import relu


class CommonBlock(nn.Module):
    def __init__(self, output_channel, block_channel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(output_channel, block_channel, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            block_channel, block_channel, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(block_channel, output_channel, kernel_size=1, stride=1)

    def forward(self, x):
        return relu(self.conv3(self.conv2(self.conv1(x))) + x)


class BlockRep(nn.Module):
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
        self.upsampling_layer = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=3,
            stride=first_layer_center_conv_stride,
            padding=1,
        )
        self.b1_conv1 = nn.Conv2d(input_channel, block_channel, kernel_size=1, stride=1)
        self.b1_conv2 = nn.Conv2d(
            block_channel,
            block_channel,
            kernel_size=3,
            stride=first_layer_center_conv_stride,
            padding=1,
        )
        self.b1_conv3 = nn.Conv2d(
            block_channel, output_channel, kernel_size=1, stride=1
        )

        self.seq = nn.Sequential()
        for _ in range(1, repetitions):
            self.seq.append(CommonBlock(output_channel, block_channel))

    def forward(self, x):
        upsampled = self.upsampling_layer(x)
        x = torch.relu(self.b1_conv3(self.b1_conv2(self.b1_conv1(x))) + upsampled)
        return self.seq(x)


def main():
    x = torch.randn((1, 3, 224, 224))

    conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    x = maxpool(conv(x))

    first_layer_center_conv_stride = 1
    input_channel = 64
    block_channel = 64
    output_channel = block_channel * 4
    # block channel

    b = BlockRep(
        input_channel,
        block_channel,
        output_channel,
        first_layer_center_conv_stride,
        repetitions=3,
    )

    x = b(x)
    print(x.shape)

    first_layer_center_conv_stride = 2
    input_channel = block_channel * 4  # 64 * 4
    block_channel = block_channel * 2  # 64 * 2
    output_channel = block_channel * 4  # 64 * 2 * 4

    b = BlockRep(
        input_channel,
        block_channel,
        output_channel,
        first_layer_center_conv_stride,
        repetitions=4,
    )

    x = b(x)
    print(x.shape)


if __name__ == "__main__":
    main()
