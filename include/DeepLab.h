#pragma once
#include"ResNet.h"
#include"DeepLabDecoder.h"

class DeepLabV3Impl : public torch::nn::Module
{
public:
	DeepLabV3Impl(int num_classes, std::string encoder_name = "resnet18", std::string pretrained_path = "", int encoder_depth = 5,
		int decoder_channels = 256, int in_channels = 3, double upsampling = 8);
	torch::Tensor forward(torch::Tensor x);
private:
	ResNet encoder{ nullptr };
	DeepLabV3Decoder decoder{ nullptr };
	SegmentationHead segmentation_head{ nullptr };
	int num_classes = 1;
	std::vector<int> BasicChannels = { 3, 64, 64, 128, 256, 512 };
	std::vector<int> BottleChannels = { 3, 64, 256, 512, 1024, 2048 };
	std::map<std::string, std::vector<int>> name2layers = getParams();
}; TORCH_MODULE(DeepLabV3);

class DeepLabV3PlusImpl : public torch::nn::Module
{
public:
	DeepLabV3PlusImpl(int num_classes, std::string encoder_name = "resnet18", std::string pretrained_path = "", int encoder_depth = 5,
		int encoder_output_stride = 16, int decoder_channels = 256, int in_channels = 3, double upsampling = 4);
	torch::Tensor forward(torch::Tensor x);
private:
	ResNet encoder{ nullptr };
	DeepLabV3PlusDecoder decoder{ nullptr };
	SegmentationHead segmentation_head{ nullptr };
	int num_classes = 1;
	std::vector<int> BasicChannels = { 3, 64, 64, 128, 256, 512 };
	std::vector<int> BottleChannels = { 3, 64, 256, 512, 1024, 2048 };
	std::map<std::string, std::vector<int>> name2layers = getParams();
	std::vector<int> decoder_atrous_rates = { 12, 24, 36 };
}; TORCH_MODULE(DeepLabV3Plus);