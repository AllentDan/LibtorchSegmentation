#pragma once
#include"../utils/util.h"

torch::nn::Sequential TransposeX2(int in_channels, int out_channels, bool use_batchnorm = true);

torch::nn::Sequential Conv2dReLU(int in_channels, int out_channels, int kernel_size, int padding = 0,
	int stride = 1, bool use_batchnorm = true);

class DecoderBlockLinkImpl : public torch::nn::Module {
public:
	DecoderBlockLinkImpl(int in_channels, int out_channels, bool use_batchnorm = true);
	torch::Tensor forward(torch::Tensor x, torch::Tensor skip);
private:
	torch::nn::Sequential conv2drelu1 { nullptr };
	torch::nn::Sequential conv2drelu2{ nullptr };
	torch::nn::Sequential transpose{ nullptr };
}; TORCH_MODULE(DecoderBlockLink);

class LinknetDecoderImpl : public torch::nn::Module
{
public:
	LinknetDecoderImpl(std::vector<int> encoder_channels, int prefinal_channels = 32,
		int n_blocks = 5, bool use_batchnorm = true);
	torch::Tensor forward(std::vector< torch::Tensor> x_list);
private:
	torch::nn::ModuleList blocks;
}; TORCH_MODULE(LinknetDecoder);

