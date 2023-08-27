import torch
import torch.nn as nn
from torch.nn.functional import relu


def main():
    x = torch.randn((1, 3, 224, 224))

    conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    x = maxpool(conv(x))

    upsampling_layer = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
    upsampled = upsampling_layer(x)
    conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
    x = torch.relu(conv3(conv2(conv1(x))) + upsampled)
    print(x.shape)
    conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)
    conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)

    upsampling_layer = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
    upsampled = upsampling_layer(x)
    conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
    conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1)
    x = torch.relu(conv3(conv2(conv1(x))) + upsampled)
    print(x.shape)
    conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)
    conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)
    conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)

    upsampling_layer = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
    upsampled = upsampling_layer(x)
    conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
    conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1)
    x = torch.relu(conv3(conv2(conv1(x))) + upsampled)
    print(x.shape)
    conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)
    conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)
    conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)
    conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)
    conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)

    upsampling_layer = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)
    upsampled = upsampling_layer(x)
    conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
    conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1)
    x = torch.relu(conv3(conv2(conv1(x))) + upsampled)
    print(x.shape)
    conv1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)
    conv1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)
    conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1)
    previous = x
    x = relu(conv3(conv2(conv1(x))) + previous)
    print(x.shape)


if __name__ == "__main__":
    main()
