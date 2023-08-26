## [PyTorch] Implementation of benchmark object classification/detection models in CPP and python.

This repository aims to help learners to compare the python and cpp api of pytorch using benchmark models.

**Implemented Models**
- **VGG** ([py](./src_py/models/vgg/model.py) | [cpp](./src_cpp/models/vgg.cpp))

#### Quickstart
[setup.py](setup.py) contains various pre-requisites needed to setup.
```sh
git clone https://github.com/akarsh1995/computer-vision-models-cpp-py.git
cd computer-vision-models-cpp-py
pipenv install
python setup.py
```

```sh
cd build
cmake ..
cmake --build . --config Release

# forward pass for VGG model
./NNet_vgg
```

let python find src_py directory as package

```
export PYTHONPATH="src_py"
```

use models
```py
import torch
from models.vgg import VGGNet

def test_model():
    model = VGGNet(in_channels=3, num_classes=1000, im_shape=(224, 224))
    assert model(model.gen_random_input()).shape == torch.Size([1, 1000])
```

