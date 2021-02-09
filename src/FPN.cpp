#include "FPN.h"

FPNImpl::FPNImpl(int _num_classes, std::string encoder_name, std::string pretrained_path, int encoder_depth,
                 int decoder_pyramid_channel, int decoder_segmentation_channels, std::string decoder_merge_policy,
                 float decoder_dropout, double upsampling){
    num_classes = _num_classes;
    std::vector<int> encoder_channels = BasicChannels;
    if(!name2layers.count(encoder_name)) throw "encoder name must in {resnet18, resnet34, resnet50, resnet101}";
    if(encoder_name!="resnet18" && encoder_name!="resnet34"){
        encoder_channels = BottleChannels;
    }
    encoder = pretrained_resnet(1000, encoder_name, pretrained_path);
    decoder = FPNDecoder(encoder_channels,encoder_depth, decoder_pyramid_channel,
                         decoder_segmentation_channels,decoder_dropout, decoder_merge_policy);
    segmentation_head = SegmentationHead(decoder_segmentation_channels,num_classes,1,upsampling);

    register_module("encoder",encoder);
    register_module("decoder",decoder);
    register_module("segmentation_head",segmentation_head);
}

torch::Tensor FPNImpl::forward(torch::Tensor x){
    std::vector<torch::Tensor> features = encoder->features(x);
    x = decoder->forward(features);
    x = segmentation_head->forward(x);
    return x;
}
