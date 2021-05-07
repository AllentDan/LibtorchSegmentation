#include "DeepLab.h"

DeepLabV3Impl::DeepLabV3Impl(int _num_classes, std::string encoder_name, std::string pretrained_path, int encoder_depth,
	int decoder_channels, int in_channels, double upsampling) {
	num_classes = _num_classes;
	auto encoder_param = encoder_params();
	std::vector<int> encoder_channels = encoder_param[encoder_name]["out_channels"];
	if (!encoder_param.contains(encoder_name))
		std::cout<< "encoder name must in {resnet18, resnet34, resnet50, resnet101, resnet150, \
				resnext50_32x4d, resnext101_32x8d, vgg11, vgg11_bn, vgg13, vgg13_bn, \
				vgg16, vgg16_bn, vgg19, vgg19_bn,}";
	if (encoder_param[encoder_name]["class_type"] == "resnet")
		encoder = new ResNetImpl(encoder_param[encoder_name]["layers"], 1000, encoder_name);
	else if (encoder_param[encoder_name]["class_type"] == "vgg")
		encoder = new VGGImpl(encoder_param[encoder_name]["cfg"], 1000, encoder_param[encoder_name]["batch_norm"]);
	else std::cout<< "unknown error in backbone initialization";

	encoder->load_pretrained(pretrained_path);
	encoder->make_dilated({ 5,4 }, {4,2});

	decoder = DeepLabV3Decoder(encoder_channels[encoder_channels.size()-1], decoder_channels);
	segmentation_head = SegmentationHead(decoder_channels, num_classes, 1, upsampling);

	register_module("encoder", std::shared_ptr<Backbone>(encoder));
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
	auto encoder_param = encoder_params();
	std::vector<int> encoder_channels = encoder_param[encoder_name]["out_channels"];
	if (!encoder_param.contains(encoder_name))
		std::cout<< "encoder name must in {resnet18, resnet34, resnet50, resnet101, resnet150, \
				resnext50_32x4d, resnext101_32x8d, vgg11, vgg11_bn, vgg13, vgg13_bn, \
				vgg16, vgg16_bn, vgg19, vgg19_bn,}";
	if (encoder_param[encoder_name]["class_type"] == "resnet")
		encoder = new ResNetImpl(encoder_param[encoder_name]["layers"], 1000, encoder_name);
	else if (encoder_param[encoder_name]["class_type"] == "vgg")
		encoder = new VGGImpl(encoder_param[encoder_name]["cfg"], 1000, encoder_param[encoder_name]["batch_norm"]);
	else std::cout<< "unknown error in backbone initialization";

	encoder->load_pretrained(pretrained_path);
	if (encoder_output_stride == 8) {
		encoder->make_dilated({ 5,4 }, { 4,2 });
	}
	else if (encoder_output_stride == 16) {
		encoder->make_dilated({ 5 }, { 2 });
	}
	else {
		std::cout<< "Encoder output stride should be 8 or 16";
	}

	decoder = DeepLabV3PlusDecoder(encoder_channels, decoder_channels, decoder_atrous_rates, encoder_output_stride);
	segmentation_head = SegmentationHead(decoder_channels, num_classes, 1, upsampling);

	register_module("encoder", std::shared_ptr<Backbone>(encoder));
	register_module("decoder", decoder);
	register_module("segmentation_head", segmentation_head);
}

torch::Tensor DeepLabV3PlusImpl::forward(torch::Tensor x) {
	std::vector<torch::Tensor> features = encoder->features(x);
	x = decoder->forward(features);
	x = segmentation_head->forward(x);
	return x;
}
