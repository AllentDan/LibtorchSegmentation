#include "PSPNet.h"

PSPNetImpl::PSPNetImpl(int _num_classes, std::string encoder_name, std::string pretrained_path, int _encoder_depth,
	int psp_out_channels, bool psp_use_batchnorm, float psp_dropout, double upsampling) {
	num_classes = _num_classes;
	encoder_depth = _encoder_depth;

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
	decoder = PSPDecoder(encoder_channels, psp_out_channels, psp_dropout, psp_use_batchnorm);
	segmentation_head = SegmentationHead(psp_out_channels, num_classes, 3, upsampling);

	register_module("encoder", std::shared_ptr<Backbone>(encoder));
	register_module("decoder", decoder);
	register_module("segmentation_head", segmentation_head);
}

torch::Tensor PSPNetImpl::forward(torch::Tensor x) {
	std::vector<torch::Tensor> features = encoder->features(x, encoder_depth);
	x = decoder->forward(features);
	x = segmentation_head->forward(x);
	return x;
}
