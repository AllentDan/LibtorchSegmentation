#include "util.h"

SegmentationHeadImpl::SegmentationHeadImpl(int in_channels, int out_channels, int kernel_size, double _upsampling){
    conv2d = torch::nn::Conv2d(conv_options(in_channels, out_channels, kernel_size, 1, kernel_size / 2));
    upsampling = torch::nn::Upsample(upsample_options(std::vector<double>{_upsampling,_upsampling}));
    register_module("conv2d",conv2d);
}
torch::Tensor SegmentationHeadImpl::forward(torch::Tensor x){
    x = conv2d->forward(x);
    x = upsampling->forward(x);
    return x;
}

std::string replace_all_distinct(std::string str, const std::string old_value, const std::string new_value)
{
    for (std::string::size_type pos(0); pos != std::string::npos; pos += new_value.length())
    {
        if ((pos = str.find(old_value, pos)) != std::string::npos)
        {
            str.replace(pos, old_value.length(), new_value);
        }
        else { break; }
    }
    return   str;
}

//遍历该目录下的.xml文件，并且找到对应的
void load_seg_data_from_folder(std::string folder, std::string image_type,
                               std::vector<std::string> &list_images, std::vector<std::string> &list_labels)
{
    for_each_file(folder,
                  [&](const char*path,const char* name){
                      auto full_path=std::string(path).append({file_sepator()}).append(name);
                      std::string lower_name=tolower1(name);

                      if(end_with(lower_name,".json")){
                          list_labels.push_back(full_path);
                          std::string image_path = replace_all_distinct(full_path, ".json", image_type);
                          list_images.push_back(image_path);
                      }
                      return false;
                  }
            ,true
    );
}


nlohmann::json encoder_params() {
	nlohmann::json params = {
	{"resnet18", {
		{"class_type", "resnet"},
		{"out_channels", {3, 64, 64, 128, 256, 512}},
		{"layers" , {2, 2, 2, 2}},
		},
	},
	{"resnet34", {
		{"class_type", "resnet"},
		{"out_channels", {3, 64, 64, 128, 256, 512}},
		{"layers" , {3, 4, 6, 3}},
		},
	},
	{"resnet50", {
		{"class_type", "resnet"},
		{"out_channels", {3, 64, 256, 512, 1024, 2048}},
		{"layers" , {3, 4, 6, 3}},
		},
	},
	{"resnet101", {
		{"class_type", "resnet"},
		{"out_channels", {3, 64, 256, 512, 1024, 2048}},
		{"layers" , {3, 4, 23, 3}},
		},
	},
	{"resnet101", {
		{"class_type", "resnet"},
		{"out_channels", {3, 64, 256, 512, 1024, 2048}},
		{"layers" , {3, 8, 36, 3}},
		},
	},
	{"resnext50_32x4d", {
		{"class_type", "resnet"},
		{"out_channels", {3, 64, 256, 512, 1024, 2048}},
		{"layers" , {3, 4, 6, 3}},
		},
	},
	{"resnext101_32x8d", {
		{"class_type", "resnet"},
		{"out_channels", {3, 64, 256, 512, 1024, 2048}},
		{"layers" , {3, 4, 23, 3}},
		},
	},
	{"vgg11", {
		{"class_type", "vgg"},
		{"out_channels", {64, 128, 256, 512, 512, 512}},
		{"cfg",{64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
		{"batch_norm" , false},
		},
	},
	{"vgg11_bn", {
		{"class_type", "vgg"},
		{"out_channels", {64, 128, 256, 512, 512, 512}},
		{"cfg",{64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
		{"batch_norm" , true},
		},
	},
	{"vgg13", {
		{"class_type", "vgg"},
		{"out_channels", {64, 128, 256, 512, 512, 512}},
		{"cfg",{64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
		{"batch_norm" , false},
		},
	},
	{"vgg13_bn", {
		{"class_type", "vgg"},
		{"out_channels", {64, 128, 256, 512, 512, 512}},
		{"cfg",{64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
		{"batch_norm" , true},
		},
	},
	{"vgg16", {
		{"class_type", "vgg"},
		{"out_channels", {64, 128, 256, 512, 512, 512}},
		{"cfg",{64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1}},
		{"batch_norm" , false},
		},
	},
	{"vgg16_bn", {
		{"class_type", "vgg"},
		{"out_channels", {64, 128, 256, 512, 512, 512}},
		{"cfg",{64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1}},
		{"batch_norm" , true},
		},
	},
	{"vgg19", {
		{"class_type", "vgg"},
		{"out_channels", {64, 128, 256, 512, 512, 512}},
		{"cfg",{64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1}},
		{"batch_norm" , false},
		},
	},
	{"vgg19_bn", {
		{"class_type", "vgg"},
		{"out_channels", {64, 128, 256, 512, 512, 512}},
		{"cfg",{64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1}},
		{"batch_norm" , true},
		},
	},
	};
	return params;
}
