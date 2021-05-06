#include "PSPNetDecoder.h"

PSPBlockImpl::PSPBlockImpl(int in_channels, int out_channels, int pool_size, bool use_bathcnorm) {
	if (pool_size == 1)
		use_bathcnorm = false;
	pool = torch::nn::Sequential(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(pool_size)));
	conv = Conv2dReLU(in_channels, out_channels, 1, 0, 1, use_bathcnorm);

	register_module("pool", pool);
	register_module("conv", conv);
}

torch::Tensor PSPBlockImpl::forward(torch::Tensor x) {
	auto h = x.sizes()[2];
	auto w = x.sizes()[3];
	x = pool->forward(x);
	x = conv->forward(x);
	x = at::upsample_bilinear2d(x, { h ,w }, true);
	return x;
}


PSPModuleImpl::PSPModuleImpl(int in_channels, std::vector<int> _sizes, bool use_bathcnorm) {
	for (auto size : _sizes) {
		blocks->push_back(PSPBlock(in_channels, in_channels / _sizes.size(), size, use_bathcnorm));
	}
	register_module("blocks", blocks);
}

torch::Tensor PSPModuleImpl::forward(torch::Tensor x) {
	std::vector<torch::Tensor> xs;
	for (int i = 0; i < blocks->size(); i++) {
		xs.push_back(blocks[i]->as<PSPBlock>()->forward(x));
	}
	xs.push_back(x);
	x = torch::cat(xs, 1);
	return x;
}

PSPDecoderImpl::PSPDecoderImpl(std::vector<int> encoder_channels, int out_channels, double _dropout,
	bool use_batchnorm, int _encoder_depth) {
	encoder_depth = _encoder_depth;
	std::vector<int> size = { 1, 2, 3, 6 };
	psp = PSPModule(encoder_channels[encoder_depth], size, use_batchnorm);
	conv = Conv2dReLU(encoder_channels[encoder_depth] * 2, out_channels, 1, 0, 1, use_batchnorm);
	dropout = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(_dropout));

	register_module("psp", psp);
	register_module("conv", conv);
}

torch::Tensor PSPDecoderImpl::forward(std::vector<torch::Tensor> features) {
	auto x = features[features.size()-1];
	x = psp->forward(x);
	x = conv->forward(x);
	x = dropout->forward(x);
	return x;
}