import torch
from models.resnet import ResNet50


def test_model():
    x = torch.randn((1, 3, 224, 224))
    model = ResNet50(repetitions=[3, 4, 6, 3])
    assert model(x).shape == torch.Size([1, 1000])
