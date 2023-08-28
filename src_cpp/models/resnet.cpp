#include <iostream>
#include <torch/torch.h>

// Define a new Module.
struct CommonBlockImpl : torch::nn::Module {
  int output_channel;
  int block_channel;

  CommonBlockImpl(int output_channel, int block_channel)
      : output_channel(output_channel), block_channel(block_channel) {

    auto seq = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(output_channel,
                                                   block_channel,
                                                   1) // kernel_size = 1
                              .stride(1)),
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(block_channel)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),

        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(block_channel, block_channel, 3)
                .stride(1)
                .padding(1)),
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(block_channel)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),

        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(block_channel, output_channel, 1)
                .stride(1)),
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(output_channel)));

    block = register_module("block", seq);
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    x += block->forward(x);
    x = torch::nn::functional::relu(x);
    return x;
  }

  torch::nn::Sequential block{nullptr};
};

TORCH_MODULE(CommonBlock);

struct BlockGroupImpl : torch::nn::Module {
  BlockGroupImpl(int input_channel, int block_channel, int output_channel,
                 int first_layer_center_conv_stride, int repetitions = 1)
      : downsampling_layer(torch::nn::Sequential(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_channel, output_channel, 1)
                    .stride(first_layer_center_conv_stride)
                    .bias(false)),
            torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(output_channel)))),

        first_block(torch::nn::Sequential(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_channel, block_channel, 1)
                    .stride(1)),
            torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(block_channel)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),

            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(block_channel, block_channel, 3)
                    .stride(first_layer_center_conv_stride)
                    .padding(1)),
            torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(block_channel)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),

            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(block_channel, output_channel, 1)
                    .stride(1)),
            torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(output_channel)))) {

    auto s = torch::nn::Sequential();

    for (int i = 1; i < repetitions; i++) {
      s->push_back(CommonBlock(output_channel, block_channel));
    }
    seq = register_module("seq", s);
    downsampling_layer =
        register_module("downsampling_layer", downsampling_layer);
    first_block = register_module("first_block", first_block);
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor downsampled = downsampling_layer->forward(x);
    x = first_block->forward(x);
    x += downsampled;
    x = torch::nn::functional::relu(x);
    x = seq->forward(x);
    return x;
  }

  torch::nn::Sequential first_block{nullptr};
  torch::nn::Sequential seq{nullptr};
  torch::nn::Sequential downsampling_layer{nullptr};
};

TORCH_MODULE(BlockGroup);

struct ResNet50Impl : torch::nn::Module {
  ResNet50Impl(std::vector<int> repetitions) {

    conv = register_module(
        "conv", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)));
    maxpool = register_module(
        "maxpool",
        torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions({3, 3}).stride(2).padding(1)));

    auto s = torch::nn::Sequential();

    int first_layer_center_conv_stride = 1; // 1, 2, 2, 2
    int input_channel = 64;                 // 64, 256, 512, 1024
    int block_channel = 64;                 // 64, 128, 256, 512
    int output_channel = block_channel * 4; // 256, 512, 1024, 2048

    for (int i = 0; i < repetitions.size(); i++) {
      s->push_back(BlockGroup(input_channel, block_channel, output_channel,
                              first_layer_center_conv_stride, repetitions[i]));

      first_layer_center_conv_stride = 2;
      input_channel = output_channel;
      block_channel = block_channel * 2;
      output_channel = block_channel * 4;
    }
    block_groups = register_module("block_groups", s);

    avgpool = register_module("avgpool",
                              torch::nn::AdaptiveAvgPool2d(
                                  torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    fc = register_module("fc", torch::nn::Linear(input_channel, 1000));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = conv(x);
    x = maxpool(x);
    x = block_groups->forward(x);
    x = avgpool(x);
    x = fc(x.flatten(1));
    x = torch::nn::functional::softmax(
        x, torch::nn::functional::SoftmaxFuncOptions(1));
    return x;
  }

  torch::nn::Conv2d conv{nullptr};
  torch::nn::MaxPool2d maxpool{nullptr};
  torch::nn::Sequential block_groups{nullptr};
  torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
  torch::nn::Linear fc{nullptr};
};

TORCH_MODULE(ResNet50);

int main() {
  auto repetitions = {3, 4, 6, 3};
  auto net = std::make_shared<ResNet50Impl>(repetitions);
  auto result = net->forward(torch::randn({1, 3, 224, 224}));
  assert(result.sizes()[1] == 1000);
}
