#pragma once
#include"ResNet.h"
#include"LinknetDecoder.h"

class LinkNetImpl : public torch::nn::Module
{
public:
	LinkNetImpl(int num_classes, std::string encoder_name = "resnet18", std::string pretrained_path = "", int encoder_depth = 5,
		int decoder_use_batchnorm = true);
	torch::Tensor forward(torch::Tensor x);
private:
	ResNet encoder{ nullptr };
	LinknetDecoder decoder{ nullptr };
	SegmentationHead segmentation_head{ nullptr };
	int num_classes = 1;
	std::vector<int> BasicChannels = { 3, 64, 64, 128, 256, 512 };
	std::vector<int> BottleChannels = { 3, 64, 256, 512, 1024, 2048 };
	std::map<std::string, std::vector<int>> name2layers = getParams();
}; TORCH_MODULE(LinkNet);


