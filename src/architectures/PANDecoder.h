#pragma once
#include"../utils/util.h"
//Pyramid Attention Network Decoder

class ConvBnReluImpl : public torch::nn::Module {
public:
	ConvBnReluImpl(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0,
		int dilation = 1, int groups = 1, bool bias = true, bool add_relu = true, bool interpolate = false);
	torch::Tensor forward(torch::Tensor x);
private:
	bool add_relu;
	bool interpolate;
	torch::nn::Conv2d conv{ nullptr };
	torch::nn::BatchNorm2d bn{ nullptr };
	torch::nn::ReLU activation{ nullptr };
	torch::nn::Upsample up{ nullptr };
}; TORCH_MODULE(ConvBnRelu);

class FPABlockImpl : public torch::nn::Module {
public:
	FPABlockImpl(int in_channels, int out_channels, std::string upscale_mode = "bilinear");
	torch::Tensor forward(torch::Tensor x);
private:
	bool align_corners;
	torch::nn::Sequential branch1{ nullptr };
	torch::nn::Sequential mid{ nullptr };
	torch::nn::Sequential down1{ nullptr };
	torch::nn::Sequential down2{ nullptr };
	torch::nn::Sequential down3{ nullptr };
	ConvBnRelu conv1{ nullptr };
	ConvBnRelu conv2{ nullptr };
}; TORCH_MODULE(FPABlock);

class GAUBlockImpl :public torch::nn::Module {
public:
	GAUBlockImpl(int in_channels, int out_channels, std::string upscale_mode = "bilinear");
	torch::Tensor forward(torch::Tensor x, torch::Tensor y);
private:
	bool align_corners;
	torch::nn::Sequential conv1{ nullptr };
	ConvBnRelu conv2{ nullptr };
}; TORCH_MODULE(GAUBlock);

class PANDecoderImpl:public torch::nn::Module
{
public:
	PANDecoderImpl(std::vector<int> encoder_channels, int decoder_channels, std::string upscale_mode = "bilinear");
	torch::Tensor forward(std::vector<torch::Tensor> x);
private:
	FPABlock fpa{ nullptr };
	GAUBlock gau3{ nullptr };
	GAUBlock gau2{ nullptr };
	GAUBlock gau1{ nullptr };

}; TORCH_MODULE(PANDecoder);

