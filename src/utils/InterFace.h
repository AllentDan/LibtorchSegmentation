#pragma once
#include<torch/torch.h>
#include<torch/script.h>

class Backbone : public torch::nn::Module
{
public:
	virtual std::vector<torch::Tensor> features(torch::Tensor x, int encoder_depth = 5) = 0;
	virtual torch::Tensor features_at(torch::Tensor x, int stage_num) = 0;
	virtual void load_pretrained(std::string pretrained_path)=0;
	virtual void make_dilated(std::vector<int> stage_list, std::vector<int> dilation_list)=0;
	virtual ~Backbone() {}
};
