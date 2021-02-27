#include "PANDecoder.h"

ConvBnReluImpl::ConvBnReluImpl(int in_channels, int out_channels, int kernel_size, int stride, int padding,
	int dilation, int groups, bool bias, bool _add_relu, bool _interpolate) {
	add_relu = _add_relu;
	interpolate = _interpolate;
	conv = torch::nn::Conv2d(conv_options(in_channels, out_channels, kernel_size, stride, padding, groups, bias, dilation));
	bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
	activation = torch::nn::ReLU(torch::nn::ReLUOptions(true));
	up = torch::nn::Upsample(upsample_options(std::vector<double>{2, 2}, true).mode(torch::kBilinear));

	register_module("conv", conv);
	register_module("bn", bn);
}

torch::Tensor ConvBnReluImpl::forward(torch::Tensor x) {
	x = conv->forward(x);
	x = bn->forward(x);
	if (add_relu)
		x = activation->forward(x);
	if (interpolate)
		x = up->forward(x);
	return x;
}

FPABlockImpl::FPABlockImpl(int in_channels, int out_channels, std::string _upscale_mode) {
	align_corners = _upscale_mode == "bilinear";
	branch1 = torch::nn::Sequential(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)),
		ConvBnRelu(in_channels, out_channels, 1, 1, 0));
	mid = torch::nn::Sequential(ConvBnRelu(in_channels, out_channels, 1, 1, 0));
	down1 = torch::nn::Sequential(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
		ConvBnRelu(in_channels, 1, 7, 1, 3));
	down2 = torch::nn::Sequential(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
		ConvBnRelu(1, 1, 5, 1, 2));
	down3 = torch::nn::Sequential(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
		ConvBnRelu(1, 1, 3, 1, 1),
		ConvBnRelu(1, 1, 3, 1, 1));
	conv2 = ConvBnRelu(1, 1, 5, 1, 2);
	conv1 = ConvBnRelu(1, 1, 7, 1, 3);

	register_module("branch1", branch1);
	register_module("mid", mid);
	register_module("down1", down1);
	register_module("down2", down2);
	register_module("down3", down3);
	register_module("conv2", conv2);
	register_module("conv1", conv1);
}

torch::Tensor FPABlockImpl::forward(torch::Tensor x) {
	auto h = x.sizes()[2];
	auto w = x.sizes()[3];
	auto b1 = branch1->forward(x);
	b1 = at::upsample_bilinear2d(b1, { h,w }, align_corners); 
	
	auto mid_tensor = mid->forward(x);
	auto x1 = down1->forward(x);
	auto x2 = down2->forward(x1);
	auto x3 = down3->forward(x2);
	x3 = at::upsample_bilinear2d(x3, {h/4,w/4}, align_corners);

	x2 = conv2->forward(x2);
	x = x2 + x3;
	x = at::upsample_bilinear2d(x, { h / 2,w / 2 }, align_corners);

	x1 = conv1->forward(x1);
	x = x + x1;
	x = at::upsample_bilinear2d(x, { h ,w }, align_corners);

	x = torch::mul(x, mid_tensor);
	x = x + b1;
	return x;
}

GAUBlockImpl::GAUBlockImpl(int in_channels, int out_channels, std::string upscale_mode) {
	align_corners = upscale_mode == "bilinear";
	conv1 = torch::nn::Sequential(
		torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)),
		ConvBnRelu(out_channels, out_channels, 1, 1, 0, 1, 1, true, false),
		torch::nn::Sigmoid()
	);
	conv2 = ConvBnRelu(in_channels, out_channels, 3, 1, 1);

	register_module("conv1", conv1);
	register_module("conv2", conv2);
}

torch::Tensor GAUBlockImpl::forward(torch::Tensor x, torch::Tensor y) {
	auto h = x.sizes()[2];
	auto w = x.sizes()[3];
	auto y_up = at::upsample_bilinear2d(y, { h ,w }, align_corners);
	x = conv2->forward(x);
	y = conv1->forward(y);
	auto z = torch::mul(x, y);
	return y_up + z;
}

PANDecoderImpl::PANDecoderImpl(std::vector<int> encoder_channels, int decoder_channels, std::string upscale_mode) {
	fpa = FPABlock(encoder_channels[encoder_channels.size() - 1], decoder_channels);
	gau3 = GAUBlock(encoder_channels[encoder_channels.size() - 2], decoder_channels, upscale_mode);
	gau2 = GAUBlock(encoder_channels[encoder_channels.size() - 3], decoder_channels, upscale_mode);
	gau1 = GAUBlock(encoder_channels[encoder_channels.size() - 4], decoder_channels, upscale_mode);

	register_module("fpa", fpa);
	register_module("gau3", gau3);
	register_module("gau2", gau2);
	register_module("gau1", gau1);
}

torch::Tensor PANDecoderImpl::forward(std::vector<torch::Tensor> features) {
	auto bottleneck = features[features.size() - 1];
	auto x5 = fpa->forward(bottleneck);	// 1 / 32
	auto x4 = gau3->forward(features[features.size() -2], x5);	//1 / 16
	auto x3 = gau2->forward(features[features.size() -3], x4);	// 1 / 8
	auto x2 = gau1->forward(features[features.size() -4], x3);	// 1 / 4

	return x2;
}