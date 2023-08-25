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

// Define a new Module.
struct Net : torch::nn::Module {
  int in_channels;
  int num_classes;
  const std::vector<int> im_shape;

  Net(int in_channels, int num_classes, std::vector<int> im_shape,
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
        ic = layer.channels;
      } else {
        seq->push_back(
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})));
      }
    }

    convs = register_module("convs", seq);

    auto input_features =
        convs->forward(torch::randn({1, in_channels, im_shape[0], im_shape[1]}))
            .flatten(1)
            .sizes()[1];

    auto fully_connected_layers = torch::nn::Sequential(
        torch::nn::Linear(input_features, 4096), torch::nn::ReLU(),
        torch::nn::Dropout(0.5), torch::nn::Linear(4096, 4096),
        torch::nn::ReLU(), torch::nn::Dropout(0.5),
        torch::nn::Linear(4096, num_classes));

    fcs = register_module("fcs", fully_connected_layers);
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    x = convs->forward(x);
    x = fcs->forward(x.flatten(1));
    return x;
  }

  torch::Tensor gen_random_input() {
    return torch::randn({1, in_channels, im_shape[0], im_shape[1]});
  }

  torch::nn::Sequential convs{nullptr}, fcs{nullptr};
};

int main() {
  // Create a new Net.
  int num_classes = 1000;
  int input_channels = 3;
  auto size = {224, 224};
  auto net = Net(input_channels, num_classes, size, VGG16);
  auto result = net.forward(net.gen_random_input());
  assert(result.sizes()[1] == num_classes);
}
