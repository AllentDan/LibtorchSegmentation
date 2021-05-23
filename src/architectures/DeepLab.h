#pragma once
#include"../backbones/ResNet.h"
#include"../backbones/VGG.h"
#include"DeepLabDecoder.h"

class DeepLabV3Impl : public torch::nn::Module
{
public:
	DeepLabV3Impl() {}
	~DeepLabV3Impl() {
		//delete encoder;
	}
	DeepLabV3Impl(int num_classes, std::string encoder_name = "resnet18", std::string pretrained_path = "", int encoder_depth = 5,
		int decoder_channels = 256, int in_channels = 3, double upsampling = 8);
	torch::Tensor forward(torch::Tensor x);
private:
	Backbone *encoder;
	DeepLabV3Decoder decoder{ nullptr };
	SegmentationHead segmentation_head{ nullptr };
	int num_classes = 1;
}; TORCH_MODULE(DeepLabV3);

class DeepLabV3PlusImpl : public torch::nn::Module
{
public:
	DeepLabV3PlusImpl() {};
	~DeepLabV3PlusImpl() {
		//delete encoder;
	}
	DeepLabV3PlusImpl(int num_classes, std::string encoder_name = "resnet18", std::string pretrained_path = "", int encoder_depth = 5,
		int encoder_output_stride = 16, int decoder_channels = 256, int in_channels = 3, double upsampling = 4);
	torch::Tensor forward(torch::Tensor x);
private:
	Backbone* encoder;
	DeepLabV3PlusDecoder decoder{ nullptr };
	SegmentationHead segmentation_head{ nullptr };
	int num_classes = 1;
	std::vector<int> decoder_atrous_rates = { 12, 24, 36 };
}; TORCH_MODULE(DeepLabV3Plus);