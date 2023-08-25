import torch
from models.vgg import VGGNet


def test_model():
    model = VGGNet(in_channels=3, num_classes=1000, im_shape=(224, 224))
    assert model(model.gen_random_input()).shape == torch.Size([1, 1000])
