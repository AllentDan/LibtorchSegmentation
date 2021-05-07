#include "PAN.h"

PANImpl::PANImpl(int _num_classes, std::string encoder_name, std::string pretrained_path, int decoder_channels,
	double upsampling) {
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
	encoder->make_dilated({ 5 }, { 2 });

	decoder = PANDecoder(encoder_channels, decoder_channels);
	segmentation_head = SegmentationHead(decoder_channels, num_classes, 3, upsampling);

	register_module("encoder", std::shared_ptr<Backbone>(encoder));
	register_module("decoder", decoder);
	register_module("segmentation_head", segmentation_head);
}

torch::Tensor PANImpl::forward(torch::Tensor x) {
	std::vector<torch::Tensor> features = encoder->features(x);
	x = decoder->forward(features);
	x = segmentation_head->forward(x);
	return x;
}
