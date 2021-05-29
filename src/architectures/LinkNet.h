/*
This libtorch implementation is writen by AllentDan.
Copyright(c) AllentDan 2021,
All rights reserved.
*/
#pragma once
#include"../backbones/ResNet.h"
#include"../backbones/VGG.h"
#include"LinknetDecoder.h"

class LinkNetImpl : public torch::nn::Module
{
public:
	LinkNetImpl() {}
	~LinkNetImpl() {
		//delete encoder;
	}
	LinkNetImpl(int num_classes, std::string encoder_name = "resnet18", std::string pretrained_path = "", int encoder_depth = 5,
		int decoder_use_batchnorm = true);
	torch::Tensor forward(torch::Tensor x);
private:
	Backbone* encoder;
	LinknetDecoder decoder{ nullptr };
	SegmentationHead segmentation_head{ nullptr };
	int num_classes = 1;
}; TORCH_MODULE(LinkNet);


