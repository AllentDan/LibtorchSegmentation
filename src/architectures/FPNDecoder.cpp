#include "FPNDecoder.h"

Conv3x3GNReLUImpl::Conv3x3GNReLUImpl(int _in_channels, int _out_channels, bool _upsample){
    upsample = _upsample;
    block = torch::nn::Sequential(torch::nn::Conv2d(conv_options(_in_channels, _out_channels, 3, 1, 1, 1, false)),
                                   torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, _out_channels)),
                                   torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    register_module("block",block);
}

torch::Tensor Conv3x3GNReLUImpl::forward(torch::Tensor x){
    x = block->forward(x);
    if (upsample){
        x = torch::nn::Upsample(upsample_options(std::vector<double>{2,2}))->forward(x);
    }
    return x;
}

FPNBlockImpl::FPNBlockImpl(int pyramid_channels, int skip_channels)
{
    skip_conv = torch::nn::Conv2d(conv_options(skip_channels, pyramid_channels,1));
    upsample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2,2})).mode(torch::kNearest));

    register_module("skip_conv",skip_conv);
}

torch::Tensor FPNBlockImpl::forward(torch::Tensor x, torch::Tensor skip){
    x = upsample->forward(x);
    skip = skip_conv(skip);
    x = x + skip;
    return x;
}

SegmentationBlockImpl::SegmentationBlockImpl(int in_channels, int out_channels, int n_upsamples)
{
    block = torch::nn::Sequential();
    block->push_back(Conv3x3GNReLU(in_channels, out_channels, bool(n_upsamples)));
    if(n_upsamples>1){
        for (int i=1; i<n_upsamples; i++) {
            block->push_back(Conv3x3GNReLU(out_channels, out_channels, true));
        }
    }
    register_module("block",block);
}

torch::Tensor SegmentationBlockImpl::forward(torch::Tensor x){
    x = block->forward(x);
    return x;
}

//vector求和
template<typename T>
T sumTensor(std::vector<T> x_list){
    if(x_list.empty()) throw "sumTensor only accept non-empty list";
    T re = x_list[0];
    for(int i = 1; i<x_list.size(); i++){
        re+=x_list[i];
    }
    return re;
}

MergeBlockImpl::MergeBlockImpl(std::string policy){
    if(policy!=policies[0] && policy!=policies[1]){
        throw "`merge_policy` must be one of: ['add', 'cat'], got "+policy;
    }
    _policy = policy;
}

torch::Tensor MergeBlockImpl::forward(std::vector<torch::Tensor> x){
    if(_policy=="add") return sumTensor(x);
    else if (_policy == "cat") return torch::cat(x, 1);
    else throw "`merge_policy` must be one of: ['add', 'cat'], got "+_policy;
}

FPNDecoderImpl::FPNDecoderImpl(std::vector<int> encoder_channels, int encoder_depth, int pyramid_channels, int segmentation_channels,
                       float dropout_, std::string merge_policy)
{
    out_channels = merge_policy == "add"? segmentation_channels :segmentation_channels * 4;
    if(encoder_depth<3) throw "Encoder depth for FPN decoder cannot be less than 3";
    std::reverse(std::begin(encoder_channels),std::end(encoder_channels));
    encoder_channels = std::vector<int> (encoder_channels.begin(),encoder_channels.begin()+encoder_depth+1);
    p5 = torch::nn::Conv2d(conv_options(encoder_channels[0], pyramid_channels, 1));
    p4 = FPNBlock(pyramid_channels, encoder_channels[1]);
    p3 = FPNBlock(pyramid_channels, encoder_channels[2]);
    p2 = FPNBlock(pyramid_channels, encoder_channels[3]);
    for(int i = 3; i>=0; i--){
        seg_blocks->push_back(SegmentationBlock(pyramid_channels, segmentation_channels, i));
    }
    merge = MergeBlock(merge_policy);
    dropout = torch::nn::Dropout2d(torch::nn::Dropout2dOptions().p(dropout_).inplace(true));

    register_module("p5",p5);
    register_module("p4",p4);
    register_module("p3",p3);
    register_module("p2",p2);
    register_module("seg_blocks",seg_blocks);
    register_module("merge",merge);
}

torch::Tensor FPNDecoderImpl::forward(std::vector<torch::Tensor> features){
	int features_len = features.size();
    auto _p5 = p5->forward(features[features_len-1]);
    auto _p4 = p4->forward(_p5, features[features_len - 2]);
    auto _p3 = p3->forward(_p4, features[features_len - 3]);
    auto _p2 = p2->forward(_p3, features[features_len - 4]);
    _p5 = seg_blocks[0]->as<SegmentationBlock>()->forward(_p5);
    _p4 = seg_blocks[1]->as<SegmentationBlock>()->forward(_p4);
    _p3 = seg_blocks[2]->as<SegmentationBlock>()->forward(_p3);
    _p2 = seg_blocks[3]->as<SegmentationBlock>()->forward(_p2);

    auto x = merge->forward(std::vector<torch::Tensor>{_p5,_p4,_p3,_p2});
    x = dropout->forward(x);
    return x;
}
