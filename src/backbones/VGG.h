#pragma once
#include"../utils/util.h"
#include"../utils/InterFace.h"
//对应pytorch中的make_features函数，返回CNN主体，该主体是一个torch::nn::Sequential对象
torch::nn::Sequential make_features(std::vector<int> &cfg, bool batch_norm);

//VGG类的声明，包括初始化和前向传播
class VGGImpl : public Backbone
{
private:
	torch::nn::Sequential features_{ nullptr };
	torch::nn::AdaptiveAvgPool2d avgpool{ nullptr };
	torch::nn::Sequential classifier;
	std::vector<int> cfg = {};
	bool batch_norm;

public:
	VGGImpl(std::vector<int> cfg, int num_classes = 1000, bool batch_norm = false);
	torch::Tensor forward(torch::Tensor x);

	std::vector<torch::Tensor> features(torch::Tensor x, int encoder_depth = 5) override;
	torch::Tensor features_at(torch::Tensor x, int stage_num) override;
	void make_dilated(std::vector<int> stage_list, std::vector<int> dilation_list) override;
	void load_pretrained(std::string pretrained_path) override;
};
TORCH_MODULE(VGG);
