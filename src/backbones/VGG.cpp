#include "VGG.h"

torch::nn::Sequential make_features(std::vector<int> &cfg, bool batch_norm) {
	torch::nn::Sequential features;
	int in_channels = 3;
	for (auto v : cfg) {
		if (v == -1) {
			features->push_back(torch::nn::MaxPool2d(maxpool_options(2, 2)));
		}
		else {
			auto conv2d = torch::nn::Conv2d(conv_options(in_channels, v, 3, 1, 1));
			features->push_back(conv2d);
			if (batch_norm) {
				features->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(v)));
			}
			features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
			in_channels = v;
		}
	}
	return features;
}

VGGImpl::VGGImpl(std::vector<int> _cfg, int num_classes, bool batch_norm_) {
	cfg = _cfg;
	batch_norm = batch_norm_;
	features_ = make_features(cfg, batch_norm);
	avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
	classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)));
	classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	classifier->push_back(torch::nn::Dropout());
	classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)));
	classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	classifier->push_back(torch::nn::Dropout());
	classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes)));

	features_ = register_module("features", features_);
	classifier = register_module("classifier", classifier);
}

torch::Tensor VGGImpl::forward(torch::Tensor x) {
	x = features_->forward(x);
	x = avgpool(x);
	x = torch::flatten(x, 1);
	x = classifier->forward(x);
	return torch::log_softmax(x, 1);
}


std::vector<torch::Tensor> VGGImpl::features(torch::Tensor x, int encoder_depth) {
	std::vector<torch::Tensor> ans;

	int j = 0;// layer index of features_
	for (int i = 0; i < cfg.size(); i++) {
		if (cfg[i] == -1) {
			ans.push_back(x);
			if (ans.size() == encoder_depth )
			{
				break;
			}
			x = this->features_[j++]->as<torch::nn::MaxPool2d>()->forward(x);
		}
		else {
			x = this->features_[j++]->as<torch::nn::Conv2d>()->forward(x);
			if (batch_norm) {
				x = this->features_[j++]->as<torch::nn::BatchNorm2d>()->forward(x);
			}
			x = this->features_[j++]->as<torch::nn::ReLU>()->forward(x);
		}
	}
	if (ans.size() == encoder_depth && encoder_depth==5)
	{
		x = this->features_[j++]->as<torch::nn::MaxPool2d>()->forward(x);
		ans.push_back(x);
	}
	return ans;
}

torch::Tensor VGGImpl::features_at(torch::Tensor x, int stage_num) {
	assert(stage_num > 0 && stage_num <=5 && "the stage number must in range[1,5]");
	int j = 0;
	int stage_count = 0;
	for (int i = 0; i < cfg.size(); i++) {
		if (cfg[i] == -1) {
			x = this->features_[j++]->as<torch::nn::MaxPool2d>()->forward(x);
			stage_count++;
			if (stage_count == stage_num)
				return x;
		}
		else {
			x = this->features_[j++]->as<torch::nn::Conv2d>()->forward(x);
			if (batch_norm) {
				x = this->features_[j++]->as<torch::nn::BatchNorm2d>()->forward(x);
			}
			x = this->features_[j++]->as<torch::nn::ReLU>()->forward(x);
		}
	}
	return x;
}

void VGGImpl::load_pretrained(std::string pretrained_path) {
	VGG net_pretrained = VGG(cfg, 1000, batch_norm);
	torch::load(net_pretrained, pretrained_path);

	torch::OrderedDict<std::string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
	torch::OrderedDict<std::string, at::Tensor> model_dict = this->named_parameters();

	for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
	{
		if (strstr((*n).key().data(), "classifier.")) {
			continue;
		}
		model_dict[(*n).key()] = (*n).value();
	}

	torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
	auto new_params = model_dict; // implement this
	auto params = this->named_parameters(true /*recurse*/);
	auto buffers = this->named_buffers(true /*recurse*/);
	for (auto& val : new_params) {
		auto name = val.key();
		auto* t = params.find(name);
		if (t != nullptr) {
			t->copy_(val.value());
		}
		else {
			t = buffers.find(name);
			if (t != nullptr) {
				t->copy_(val.value());
			}
		}
	}
	torch::autograd::GradMode::set_enabled(true);
	return;
}

void VGGImpl::make_dilated(std::vector<int> stage_list, std::vector<int> dilation_list) {
	std::cout<< "'VGG' models do not support dilated mode due to Max Pooling operations for downsampling!";
	return;
}
