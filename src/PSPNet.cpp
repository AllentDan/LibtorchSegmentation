#include "PSPNet.h"

PSPNetImpl::PSPNetImpl(int _num_classes, std::string encoder_name, std::string pretrained_path, int _encoder_depth,
	int psp_out_channels, bool psp_use_batchnorm, float psp_dropout, double upsampling) {
	num_classes = _num_classes;
	encoder_depth = _encoder_depth;
	std::vector<int> encoder_channels = BasicChannels;
	if (!name2layers.count(encoder_name)) throw "encoder name must in {resnet18, resnet34, resnet50, resnet101}";
	if (encoder_name != "resnet18" && encoder_name != "resnet34") {
		encoder_channels = BottleChannels;
	}
	encoder = pretrained_resnet(1000, encoder_name, pretrained_path);
	decoder = PSPDecoder(encoder_channels, psp_out_channels, psp_dropout, psp_use_batchnorm);
	segmentation_head = SegmentationHead(psp_out_channels, num_classes, 3, upsampling);

	register_module("encoder", encoder);
	register_module("decoder", decoder);
	register_module("segmentation_head", segmentation_head);
}

torch::Tensor PSPNetImpl::forward(torch::Tensor x) {
	std::vector<torch::Tensor> features = encoder->features(x, encoder_depth);
	x = decoder->forward(features);
	x = segmentation_head->forward(x);
	return x;
}
