/*
This libtorch implementation is writen by AllentDan.
Copyright(c) AllentDan 2021,
All rights reserved.
*/

#ifndef FPN_H
#define FPN_H
#include"../backbones/ResNet.h"
#include"../backbones/VGG.h"
#include"FPNDecoder.h"

class FPNImpl : public torch::nn::Module
{
public:
	FPNImpl() {}
	~FPNImpl() { 
		//delete encoder;
	}
    FPNImpl(int num_classes, std::string encoder_name = "resnet18", std::string pretrained_path = "", int encoder_depth = 5,
            int decoder_pyramid_channel=256, int decoder_segmentation_channels = 128, std::string decoder_merge_policy = "add",
            float decoder_dropout = 0.2, double upsampling = 4);
    torch::Tensor forward(torch::Tensor x);
private:
    Backbone *encoder;
    FPNDecoder decoder{nullptr};
    SegmentationHead segmentation_head{nullptr};
    int num_classes = 1;
};TORCH_MODULE(FPN);

#endif // FPN_H
