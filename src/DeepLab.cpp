#include "DeepLab.h"

DeepLabV3Impl::DeepLabV3Impl(int _num_classes, std::string encoder_name, std::string pretrained_path, int encoder_depth,
	int decoder_channels, int in_channels, double upsampling) {
	num_classes = _num_classes;
	std::vector<int> encoder_channels = BasicChannels;
	if (!name2layers.count(encoder_name)) throw "encoder name must in {resnet18, resnet34, resnet50, resnet101}";
	if (encoder_name != "resnet18" && encoder_name != "resnet34") {
		encoder_channels = BottleChannels;
	}
	encoder = pretrained_resnet(1000, encoder_name, pretrained_path);
	encoder->make_dilated({ 5,4 }, {4,2});

	decoder = DeepLabV3Decoder(encoder_channels[encoder_channels.size()-1], decoder_channels);
	segmentation_head = SegmentationHead(decoder_channels, num_classes, 1, upsampling);

	register_module("encoder", encoder);
	register_module("decoder", decoder);
	register_module("segmentation_head", segmentation_head);
}

torch::Tensor DeepLabV3Impl::forward(torch::Tensor x) {
	std::vector<torch::Tensor> features = encoder->features(x);
	x = decoder->forward(features);
	x = segmentation_head->forward(x);
	return x;
}

DeepLabV3PlusImpl::DeepLabV3PlusImpl(int _num_classes, std::string encoder_name, std::string pretrained_path, int encoder_depth,
	int encoder_output_stride, int decoder_channels, int in_channels, double upsampling) {
	num_classes = _num_classes;
	std::vector<int> encoder_channels = BasicChannels;
	if (!name2layers.count(encoder_name)) throw "encoder name must in {resnet18, resnet34, resnet50, resnet101}";
	if (encoder_name != "resnet18" && encoder_name != "resnet34") {
		encoder_channels = BottleChannels;
	}
	encoder = pretrained_resnet(1000, encoder_name, pretrained_path);
	if (encoder_output_stride == 8) {
		encoder->make_dilated({ 5,4 }, { 4,2 });
	}
	else if (encoder_output_stride == 16) {
		encoder->make_dilated({ 5 }, { 2 });
	}
	else {
		throw "Encoder output stride should be 8 or 16";
	}

	decoder = DeepLabV3PlusDecoder(encoder_channels, decoder_channels, decoder_atrous_rates, encoder_output_stride);
	segmentation_head = SegmentationHead(decoder_channels, num_classes, 1, upsampling);

	register_module("encoder", encoder);
	register_module("decoder", decoder);
	register_module("segmentation_head", segmentation_head);
}

torch::Tensor DeepLabV3PlusImpl::forward(torch::Tensor x) {
	std::vector<torch::Tensor> features = encoder->features(x);
	x = decoder->forward(features);
	x = segmentation_head->forward(x);
	return x;
}
