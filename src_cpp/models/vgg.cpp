#include <iostream>
#include <torch/torch.h>

// Define a new Module.
struct Net : torch::nn::Module {
  Net(int in_channels, int num_classes, const std::vector<int> &im_shape) {
    convs = register_module(
        "convs",
        torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3)
                                  .stride(1)
                                  .padding(1)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})),

            // 2
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})),

            // 3
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})),

            // 4
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2})),

            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}))));

    auto input_features =
        convs->forward(torch::randn({1, in_channels, im_shape[0], im_shape[1]}))
            .flatten(1)
            .sizes()[1];

    fcs = register_module(
        "fcs", torch::nn::Sequential(torch::nn::Linear(input_features, 4096),
                                     torch::nn::ReLU(), torch::nn::Dropout(0.5),
                                     torch::nn::Linear(4096, 4096),
                                     torch::nn::ReLU(), torch::nn::Dropout(0.5),
                                     torch::nn::Linear(4096, num_classes)));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    x = convs->forward(x);
    x = fcs->forward(x.flatten(1));
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Sequential convs{nullptr}, fcs{nullptr};
};

int main() {
  // Create a new Net.
  int num_classes = 1000;
  int input_channels = 3;
  auto size = {224, 224};
  auto net = Net(input_channels, num_classes, size);
  auto result = net.forward(torch::randn({1, 3, 224, 224}));
  std::cout << result.sizes() << std::endl;
  assert(result.sizes()[1] == num_classes);
}
