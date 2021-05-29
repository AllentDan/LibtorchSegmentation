#ifndef SEGDATASET_H
#define SEGDATASET_H
#include"utils/util.h"
#include"fstream"
#include<opencv2/opencv.hpp>

//freeze_epochs (unsigned int): freeze the backbone during the first freeze_epochs, default 0;
//decay_epochs (std::vector<unsigned int>): every decay_epoch, learning rate will decay by 90 percent, default {0};
//dice_ce_ratio (float): the weight of dice loss in combind loss, default 0.5;
//horizontal_flip_prob (float): probability to do horizontal flip augmentation, default 0;
//vertical_flip_prob (float): probability to do vertical flip augmentation, default 0;
//scale_rotate_prob (float): probability to do rotate and scale augmentation, default 0;
struct trainTricks {
	unsigned int freeze_epochs = 0;
	std::vector<unsigned int> decay_epochs = { 0 };
	float dice_ce_ratio = 0.5;

	float horizontal_flip_prob = 0;
	float vertical_flip_prob = 0;
	float scale_rotate_prob = 0;

	float scale_limit = 0.1;
	float rotate_limit = 45;
	int interpolation = cv::INTER_LINEAR;
	int border_mode = cv::BORDER_CONSTANT;
};

void show_mask(std::string json_path, std::string image_type = ".jpg");

class SegDataset :public torch::data::Dataset<SegDataset>
{
public:
    SegDataset(int resize_width, int resize_height, std::vector<std::string> list_images,
               std::vector<std::string> list_labels, std::vector<std::string> name_list,
			   trainTricks tricks, bool isTrain = false);
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override;
    // Return the length of data
    torch::optional<size_t> size() const override {
        return list_labels.size();
    };
private:
    void draw_mask(std::string json_path, cv::Mat &mask);
	int resize_width = 512; int resize_height = 512; bool isTrain = false;
    std::vector<std::string> name_list = {};
    std::map<std::string, int> name2index = {};
    std::map<std::string, cv::Scalar> name2color = {};
    std::vector<std::string> list_images;
    std::vector<std::string> list_labels;
	trainTricks tricks;
};

#endif // SEGDATASET_H
