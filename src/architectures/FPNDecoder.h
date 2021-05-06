#ifndef FPNDECODER_H
#define FPNDECODER_H
#include"../utils/util.h"

class Conv3x3GNReLUImpl : public torch::nn::Module
{
public:
    Conv3x3GNReLUImpl(int in_channels, int out_channels, bool upsample=false);
    torch::Tensor forward(torch::Tensor x);
private:
    bool upsample;
    torch::nn::Sequential block{nullptr};
};
TORCH_MODULE(Conv3x3GNReLU);

class FPNBlockImpl : public torch::nn::Module
{
public:
    FPNBlockImpl(int pyramid_channels, int skip_channels);
    torch::Tensor forward(torch::Tensor x, torch::Tensor skip);
private:
    torch::nn::Conv2d skip_conv{nullptr};
    torch::nn::Upsample upsample{nullptr};
};
TORCH_MODULE(FPNBlock);

class SegmentationBlockImpl: public torch::nn::Module
{
public:
    SegmentationBlockImpl(int in_channels, int out_channels, int n_upsamples=0);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Sequential block{nullptr};
};TORCH_MODULE(SegmentationBlock);

class MergeBlockImpl: public torch::nn::Module{
public:
    MergeBlockImpl(std::string policy);
    torch::Tensor forward(std::vector<torch::Tensor> x);
private:
    std::string _policy;
    std::string policies[2] = {"add","cat"};
};TORCH_MODULE(MergeBlock);

class FPNDecoderImpl: public torch::nn::Module
{
public:
    FPNDecoderImpl(std::vector<int> encoder_channels = {3, 64, 64, 128, 256, 512}, int encoder_depth=5, int pyramid_channels=256,
               int segmentation_channels=128,float dropout=0.2, std::string merge_policy="add");
    torch::Tensor forward(std::vector<torch::Tensor> features);
private:
    int out_channels;
    torch::nn::Conv2d p5{nullptr};
    FPNBlock p4{nullptr};
    FPNBlock p3{nullptr};
    FPNBlock p2{nullptr};
    torch::nn::ModuleList seg_blocks{};
    MergeBlock merge{nullptr};
    torch::nn::Dropout2d dropout{nullptr};

};TORCH_MODULE(FPNDecoder);

#endif // FPNDECODER_H
