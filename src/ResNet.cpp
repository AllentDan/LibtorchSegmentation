#include "ResNet.h"

BlockImpl::BlockImpl(int64_t inplanes, int64_t planes, int64_t stride_,
    torch::nn::Sequential downsample_, bool _is_basic)
{
    downsample = downsample_;
    stride = stride_;

    conv1 = torch::nn::Conv2d(conv_options(inplanes, planes, 3, stride_, 1, false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));
    conv2 = torch::nn::Conv2d(conv_options(planes, planes, 3, 1, 1,false));
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));
    is_basic = _is_basic;
    if (!is_basic) {
        conv1 = torch::nn::Conv2d(conv_options(inplanes, planes, 1, 1, 0, false));
        conv2 = torch::nn::Conv2d(conv_options(planes, planes, 3, stride_, 1, false));
        conv3 = torch::nn::Conv2d(conv_options(planes, planes * 4, 1, 1, 0, false));
        bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes * 4));
    }

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    if (!is_basic) {
        register_module("conv3", conv3);
        register_module("bn3", bn3);
    }

    if (!downsample->is_empty()) {
        register_module("downsample", downsample);
    }
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);

    if (!is_basic) {
        x = torch::relu(x);
        x = conv3->forward(x);
        x = bn3->forward(x);
    }

    if (!downsample->is_empty()) {
        residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
}

ResNetImpl::ResNetImpl(std::vector<int> layers, int num_classes, std::string model_type)
{
    if (model_type != "resnet18" && model_type != "resnet34")
    {
        expansion = 4;
        is_basic = false;
    }
    conv1 = torch::nn::Conv2d(conv_options(3, 64, 7, 2, 3, false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
    layer1 = torch::nn::Sequential(_make_layer(64, layers[0]));
    layer2 = torch::nn::Sequential(_make_layer(128, layers[1], 2));
    layer3 = torch::nn::Sequential(_make_layer(256, layers[2], 2));
    layer4 = torch::nn::Sequential(_make_layer(512, layers[3], 2));

    fc = torch::nn::Linear(512 * expansion, num_classes);
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);
}


torch::Tensor  ResNetImpl::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = torch::avg_pool2d(x, 7, 1);
    x = x.view({ x.sizes()[0], -1 });
    x = fc->forward(x);

    return torch::log_softmax(x, 1);
}

std::vector<torch::Tensor> ResNetImpl::features(torch::Tensor x){
    std::vector<torch::Tensor> features;
    features.push_back(x);
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    features.push_back(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    features.push_back(x);
    x = layer2->forward(x);
    features.push_back(x);
    x = layer3->forward(x);
    features.push_back(x);
    x = layer4->forward(x);
    features.push_back(x);

    return features;
}

torch::nn::Sequential ResNetImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride) {

    torch::nn::Sequential downsample;
    if (stride != 1 || inplanes != planes * expansion) {
        downsample = torch::nn::Sequential(
            torch::nn::Conv2d(conv_options(inplanes, planes *  expansion, 1, stride, 0, false)),
            torch::nn::BatchNorm2d(planes *  expansion)
        );
    }
    torch::nn::Sequential layers;
    layers->push_back(Block(inplanes, planes, stride, downsample, is_basic));
    inplanes = planes *  expansion;
    for (int64_t i = 1; i < blocks; i++) {
        layers->push_back(Block(inplanes, planes, 1, torch::nn::Sequential(),is_basic));
    }

    return layers;
}

ResNet resnet18(int64_t num_classes) {
    std::vector<int> layers = { 2, 2, 2, 2 };
    ResNet model(layers, num_classes, "resnet18");
    return model;
}

ResNet resnet34(int64_t num_classes) {
    std::vector<int> layers = { 3, 4, 6, 3 };
    ResNet model(layers, num_classes, "resnet34");
    return model;
}

ResNet resnet50(int64_t num_classes) {
    std::vector<int> layers = { 3, 4, 6, 3 };
    ResNet model(layers, num_classes, "resnet50");
    return model;
}

ResNet resnet101(int64_t num_classes) {
    std::vector<int> layers = { 3, 4, 23, 3 };
    ResNet model(layers, num_classes, "resnet101");
    return model;
}

ResNet pretrained_resnet(int64_t num_classes, std::string model_name, std::string weight_path){
    std::map<std::string, std::vector<int>> name2layers = getParams();
    ResNet net_pretrained = ResNet(name2layers[model_name],1000,model_name);
    torch::load(net_pretrained, weight_path);
    if(num_classes == 1000) return net_pretrained;
    ResNet module = ResNet(name2layers[model_name],num_classes,model_name);

    torch::OrderedDict<std::string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
    torch::OrderedDict<std::string, at::Tensor> model_dict = module->named_parameters();

    for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
    {
        if (strstr((*n).key().data(), "fc.")) {
            continue;
        }
        model_dict[(*n).key()] = (*n).value();
    }

    torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
    auto new_params = model_dict; // implement this
    auto params = module->named_parameters(true /*recurse*/);
    auto buffers = module->named_buffers(true /*recurse*/);
    for (auto& val : new_params) {
        auto name = val.key();
        auto* t = params.find(name);
        if (t != nullptr) {
            t->copy_(val.value());
        }
        else {
            t = buffers.find(name);
            if (t != nullptr) {
                t->copy_(val.value());
            }
        }
    }
    torch::autograd::GradMode::set_enabled(true);
    return module;
}
