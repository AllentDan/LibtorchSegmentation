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
            // filter函数，lambda表达式
                  [&](const char*path,const char* name){
                      auto full_path=std::string(path).append({file_sepator()}).append(name);
                      std::string lower_name=tolower1(name);

                      //判断是否为jpeg文件
                      if(end_with(lower_name,".json")){
                          list_labels.push_back(full_path);
                          std::string image_path = replace_all_distinct(full_path, ".json", image_type);
                          list_images.push_back(image_path);
                      }
                      //因为文件已经已经在lambda表达式中处理了，
                      //不需要for_each_file返回文件列表，所以这里返回false
                      return false;
                  }
            ,true//递归子目录
    );
}
