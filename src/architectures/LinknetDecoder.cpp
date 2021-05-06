#include "LinknetDecoder.h"

torch::nn::Sequential TransposeX2(int in_channels, int out_channels, bool use_batchnorm) {
	torch::nn::Sequential seq = torch::nn::Sequential();
	seq->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in_channels, out_channels,4).stride(2).padding(1)));
	if (use_batchnorm)
		seq->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));
	seq->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	return seq;
}

torch::nn::Sequential Conv2dReLU(int in_channels, int out_channels, int kernel_size, int padding,
	int stride, bool use_batchnorm) {
	torch::nn::Sequential seq = torch::nn::Sequential();
	seq->push_back(torch::nn::Conv2d(conv_options(in_channels, out_channels, kernel_size, 
		stride, padding, 1, !use_batchnorm, 1)));
	if (use_batchnorm)
		seq->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));

	seq->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	return seq;
}

DecoderBlockLinkImpl::DecoderBlockLinkImpl(int in_channels, int out_channels, bool use_batchnorm) {
	conv2drelu1 = Conv2dReLU(in_channels, in_channels / 4, 1, 0, 1, use_batchnorm);
	transpose = TransposeX2(in_channels / 4, in_channels / 4, use_batchnorm);
	conv2drelu2 = Conv2dReLU(in_channels / 4, out_channels, 1, 0, 1, use_batchnorm);

	register_module("conv2drelu1", conv2drelu1);
	register_module("transpose", transpose);
	register_module("conv2drelu2", conv2drelu2);
}

torch::Tensor DecoderBlockLinkImpl::forward(torch::Tensor x, torch::Tensor skip) {
	x = conv2drelu1->forward(x);
	x = transpose->forward(x);
	x = conv2drelu2->forward(x);
	if (skip.sizes()==x.sizes())
		x = x + skip;
	return x;
}

LinknetDecoderImpl::LinknetDecoderImpl(std::vector<int> encoder_channels, int prefinal_channels,
	int n_blocks, bool use_batchnorm) {
	encoder_channels = std::vector<int>(encoder_channels.begin()+1, encoder_channels.end());
	std::reverse(std::begin(encoder_channels), std::end(encoder_channels));
	std::vector<int> channels = encoder_channels;
	channels.push_back(prefinal_channels);
	for (int i = 0; i < n_blocks; i++) {
		blocks->push_back(DecoderBlockLink(channels[i], channels[i + 1], use_batchnorm));
	}

	register_module("blocks", blocks);
}

torch::Tensor LinknetDecoderImpl::forward(std::vector< torch::Tensor> features) {
	features = std::vector<torch::Tensor>(features.begin() + 1, features.end());
	std::reverse(std::begin(features), std::end(features));

	auto x = features[0];
	auto skips = std::vector<torch::Tensor>(features.begin() + 1, features.end());
	for (int i = 0; i < blocks->size(); i++) {
		auto skip = i < skips.size() ? skips[i] : torch::zeros({ 1 });
		x = blocks[i]->as<DecoderBlockLink>()->forward(x, skip);
	}
	return x;
}