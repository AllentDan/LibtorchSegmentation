#include "DeepLabDecoder.h"


torch::nn::Sequential ASPPConv(int in_channels, int out_channels, int dilation) {
	return torch::nn::Sequential(
		torch::nn::Conv2d(conv_options(in_channels, out_channels, 3, 1, dilation, 1, false, dilation)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)),
		torch::nn::ReLU()
	);
}

torch::nn::Sequential SeparableConv2d(int in_channels, int out_channels, int kernel_size, int stride,
	int padding, int dilation, bool bias) {
	torch::nn::Conv2d dephtwise_conv = torch::nn::Conv2d(conv_options(in_channels, in_channels, kernel_size,
		stride, padding, 1, false, dilation));
	torch::nn::Conv2d pointwise_conv = torch::nn::Conv2d(conv_options(in_channels, out_channels, 1, 1, 0, 1, bias));
	return torch::nn::Sequential(dephtwise_conv, pointwise_conv);
};

torch::nn::Sequential ASPPSeparableConv(int in_channels, int out_channels, int dilation) {
	torch::nn::Sequential seq = SeparableConv2d(in_channels, out_channels, 3, 1, dilation, dilation, false);
	seq->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));
	seq->push_back(torch::nn::ReLU());
	return seq;
}

ASPPPoolingImpl::ASPPPoolingImpl(int in_channels, int out_channels) {
	seq = torch::nn::Sequential(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)),
								torch::nn::Conv2d(conv_options(in_channels, out_channels, 1, 1, 0, 1, false)),
								torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)),
								torch::nn::ReLU());
	register_module("seq", seq);
}

torch::Tensor ASPPPoolingImpl::forward(torch::Tensor x) {
	auto residual(x.clone());
	x = seq->forward(x);
	x = at::upsample_bilinear2d(x, residual[0][0].sizes(), false);
	return x;
}

ASPPImpl::ASPPImpl(int in_channels, int out_channels, std::vector<int> atrous_rates, bool separable) {
	modules->push_back(torch::nn::Sequential(torch::nn::Conv2d(conv_options(in_channels, out_channels, 1, 1, 0, 1, false)),
						torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)),
						torch::nn::ReLU()));
	if (atrous_rates.size() != 3) std::cout<< "size of atrous_rates must be 3";
	if (separable) {
		modules->push_back(ASPPSeparableConv(in_channels, out_channels, atrous_rates[0]));
		modules->push_back(ASPPSeparableConv(in_channels, out_channels, atrous_rates[1]));
		modules->push_back(ASPPSeparableConv(in_channels, out_channels, atrous_rates[2]));
	}
	else {
		modules->push_back(ASPPConv(in_channels, out_channels, atrous_rates[0]));
		modules->push_back(ASPPConv(in_channels, out_channels, atrous_rates[1]));
		modules->push_back(ASPPConv(in_channels, out_channels, atrous_rates[2]));
	}
	aspppooling = ASPPPooling(in_channels, out_channels);

	project = torch::nn::Sequential(
		torch::nn::Conv2d(conv_options(5 * out_channels, out_channels, 1, 1, 0, 1, false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)),
		torch::nn::ReLU(),
		torch::nn::Dropout(torch::nn::DropoutOptions(0.5)));

	register_module("modules", modules);
	register_module("aspppooling", aspppooling);
	register_module("project", project);
}

torch::Tensor ASPPImpl::forward(torch::Tensor x) {
	std::vector<torch::Tensor> res;
	for (int i = 0; i < 4; i++) {
		res.push_back(modules[i]->as<torch::nn::Sequential>()->forward(x));
	}
	res.push_back(aspppooling->forward(x));
	x = torch::cat(res, 1);
	x = project->forward(x);
	return x;
}

DeepLabV3DecoderImpl::DeepLabV3DecoderImpl(int in_channels, int out_channels, std::vector<int> atrous_rates) {
	seq->push_back(ASPP(in_channels, out_channels, atrous_rates));
	seq->push_back(torch::nn::Conv2d(conv_options(out_channels, out_channels, 3, 1, 1, 1, false)));
	seq->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));
	seq->push_back(torch::nn::ReLU());

	register_module("seq", seq);
}

torch::Tensor DeepLabV3DecoderImpl::forward(std::vector< torch::Tensor> x_list) {
	auto x = seq->forward(x_list[x_list.size() - 1]);
	return x;
}

DeepLabV3PlusDecoderImpl::DeepLabV3PlusDecoderImpl(std::vector<int> encoder_channels, int out_channels,
	std::vector<int> atrous_rates, int output_stride) {
	if (output_stride != 8 && output_stride != 16) std::cout<< "Output stride should be 8 or 16";
	aspp = ASPP(encoder_channels[encoder_channels.size() - 1], out_channels, atrous_rates, true);
	aspp_seq = SeparableConv2d(out_channels, out_channels, 3, 1, 1, 1, false);
	aspp_seq->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));
	aspp_seq->push_back(torch::nn::ReLU());
	double scale_factor = double(output_stride / 4);
	up = torch::nn::Upsample(torch::nn::UpsampleOptions().align_corners(true).scale_factor(std::vector<double>({ scale_factor,scale_factor })).mode(torch::kBilinear));
	int highres_in_channels = encoder_channels[encoder_channels.size() -4];
	int highres_out_channels = 48; // proposed by authors of paper

	block1 = torch::nn::Sequential(
		torch::nn::Conv2d(conv_options(highres_in_channels, highres_out_channels, 1, 1, 0, 1, false)),
		torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(highres_out_channels)),
		torch::nn::ReLU());
	block2 = SeparableConv2d(highres_out_channels + out_channels, out_channels, 3, 1, 1, 1, false);
	block2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));
	block2->push_back(torch::nn::ReLU());

	register_module("aspp", aspp);
	register_module("aspp_seq", aspp_seq);
	register_module("block1", block1);
	register_module("block2", block2);
}

torch::Tensor DeepLabV3PlusDecoderImpl::forward(std::vector<torch::Tensor> x_list) {
	auto aspp_features = aspp->forward(x_list[x_list.size() - 1]);
	aspp_features = aspp_seq->forward(aspp_features);
	aspp_features = up->forward(aspp_features);

	auto high_res_features = block1->forward(x_list[x_list.size() - 4]);
	auto concat_features = torch::cat({ aspp_features, high_res_features }, 1);
	auto fused_features = block2->forward(concat_features);
	return fused_features;
}