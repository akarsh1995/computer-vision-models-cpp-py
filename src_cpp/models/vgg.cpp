#include <iostream>
#include <torch/torch.h>

// Define a struct to represent a layer configuration
struct LayerConfig {
  int channels;        // Number of channels for convolutional layer
  bool is_max_pooling; // Indicates whether it's a max-pooling layer

  // Constructor for a convolutional layer
  LayerConfig(int channels) : channels(channels), is_max_pooling(false) {}

  // Constructor for a max-pooling layer
  LayerConfig() : channels(0), is_max_pooling(true) {}
};

const std::vector<LayerConfig> VGG16 = {
    LayerConfig(64),  LayerConfig(64),
    LayerConfig(), // Represents max-pooling
    LayerConfig(128), LayerConfig(128),
    LayerConfig(), // Represents max-pooling
    LayerConfig(256), LayerConfig(256), LayerConfig(256),
    LayerConfig(), // Represents max-pooling
    LayerConfig(512), LayerConfig(512), LayerConfig(512),
    LayerConfig(), // Represents max-pooling
    LayerConfig(512), LayerConfig(512), LayerConfig(512),
    LayerConfig() // Represents max-pooling
};

const std::vector<LayerConfig> VGG13 = {
    LayerConfig(64),  LayerConfig(64),
    LayerConfig(), // Represents max-pooling
    LayerConfig(128), LayerConfig(128),
    LayerConfig(), // Represents max-pooling
    LayerConfig(256), LayerConfig(256),
    LayerConfig(), // Represents max-pooling
    LayerConfig(512), LayerConfig(512),
    LayerConfig(), // Represents max-pooling
};

// Define a new Module.
struct NetImpl : torch::nn::Module {
  int in_channels;
  int num_classes;
  const std::vector<int> im_shape;

  NetImpl(int in_channels, int num_classes, std::vector<int> im_shape,
          std::vector<LayerConfig> vgg_map)
      : in_channels(in_channels), num_classes(num_classes), im_shape(im_shape) {

    auto seq = torch::nn::Sequential();

    auto ic = in_channels;

    for (const auto &layer : vgg_map) {
      if (!layer.is_max_pooling) {
        seq->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(ic, layer.channels, 3)
                                  .stride(1)
                                  .padding(1)));
        seq->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
        ic = layer.channels;
      } else {
        seq->push_back(
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})));
      }
    }

    avgpool = register_module("avgpool",
                              torch::nn::AdaptiveAvgPool2d(
                                  torch::nn::AdaptiveAvgPool2dOptions({7, 7})));

    features = register_module("features", seq);

    auto fully_connected_layers = torch::nn::Sequential(
        torch::nn::Linear(512 * 7 * 7, 4096),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Dropout(0.5), torch::nn::Linear(4096, 4096),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Dropout(0.5), torch::nn::Linear(4096, num_classes));

    classifier = register_module("classifier", fully_connected_layers);
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    x = features->forward(x);
    x = avgpool->forward(x);
    x = classifier->forward(x.flatten(1));
    return x;
  }

  torch::Tensor gen_random_input() {
    return torch::randn({1, in_channels, im_shape[0], im_shape[1]});
  }

  torch::nn::Sequential features{nullptr}, classifier{nullptr};
  torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
};

TORCH_MODULE(Net);

int main() {
  // Create a new Net.

  int num_classes = 1000;
  int input_channels = 3;
  auto size = {224, 224};
  auto net =
      std::make_shared<NetImpl>(input_channels, num_classes, size, VGG16);
  auto result = net->forward(net->gen_random_input());

  net = std::make_shared<NetImpl>(input_channels, num_classes, size, VGG13);
  result = net->forward(net->gen_random_input());
  assert(result.sizes()[1] == num_classes);
}
