#pragma once
/*
BSD 3 - Clause License

Copyright(c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met :

*Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This libtorch implementation is writen by AllentDan.
Copyright(c) AllentDan 2021,
All rights reserved.
*/

#include"../utils/util.h"

//struct StackSequentailImpl : torch::nn::SequentialImpl {
//	using SequentialImpl::SequentialImpl;
//
//	torch::Tensor forward(torch::Tensor x) {
//		return SequentialImpl::forward(x);
//	}
//}; TORCH_MODULE(StackSequentail);

torch::nn::Sequential ASPPConv(int in_channels, int out_channels, int dilation);

torch::nn::Sequential SeparableConv2d(int in_channels, int out_channels, int kernel_size, int stride = 1,
	int padding = 0, int dilation = 1, bool bias = true);

torch::nn::Sequential ASPPSeparableConv(int in_channels, int out_channels, int dilation);

class ASPPPoolingImpl : public torch::nn::Module {
public:
	torch::nn::Sequential seq{nullptr};
	ASPPPoolingImpl(int in_channels, int out_channels);
	torch::Tensor forward(torch::Tensor x);

}; TORCH_MODULE(ASPPPooling);

class ASPPImpl : public torch::nn::Module {
public:
	ASPPImpl(int in_channels, int out_channels, std::vector<int> atrous_rates, bool separable = false);
	torch::Tensor forward(torch::Tensor x);
private:
	torch::nn::ModuleList modules{};
	ASPPPooling aspppooling{ nullptr };
	torch::nn::Sequential project{ nullptr };
}; TORCH_MODULE(ASPP);

class DeepLabV3DecoderImpl : public torch::nn::Module
{
public:
	DeepLabV3DecoderImpl(int in_channels, int out_channels = 256, std::vector<int> atrous_rates = { 12, 24, 36 });
	torch::Tensor forward(std::vector< torch::Tensor> x);
	int out_channels = 0;
private:
	torch::nn::Sequential seq{};
}; TORCH_MODULE(DeepLabV3Decoder);

class DeepLabV3PlusDecoderImpl :public torch::nn::Module {
public:
	DeepLabV3PlusDecoderImpl(std::vector<int> encoder_channels, int out_channels,
		std::vector<int> atrous_rates, int output_stride = 16);
	torch::Tensor forward(std::vector< torch::Tensor> x);
private:
	ASPP aspp{ nullptr };
	torch::nn::Sequential aspp_seq{ nullptr };
	torch::nn::Upsample up{ nullptr };
	torch::nn::Sequential block1{ nullptr };
	torch::nn::Sequential block2{ nullptr };
}; TORCH_MODULE(DeepLabV3PlusDecoder);
