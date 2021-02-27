#include "PAN.h"

PANImpl::PANImpl(int _num_classes, std::string encoder_name, std::string pretrained_path, int decoder_channels,
	double upsampling) {
	num_classes = _num_classes;
	std::vector<int> encoder_channels = BasicChannels;
	if (!name2layers.count(encoder_name)) throw "encoder name must in {resnet18, resnet34, resnet50, resnet101}";
	if (encoder_name != "resnet18" && encoder_name != "resnet34") {
		encoder_channels = BottleChannels;
	}
	encoder = pretrained_resnet(1000, encoder_name, pretrained_path);
	encoder->make_dilated({ 5 }, { 2 });

	decoder = PANDecoder(encoder_channels, decoder_channels);
	segmentation_head = SegmentationHead(decoder_channels, num_classes, 3, upsampling);

	register_module("encoder", encoder);
	register_module("decoder", decoder);
	register_module("segmentation_head", segmentation_head);
}

torch::Tensor PANImpl::forward(torch::Tensor x) {
	std::vector<torch::Tensor> features = encoder->features(x);
	x = decoder->forward(features);
	x = segmentation_head->forward(x);
	return x;
}
