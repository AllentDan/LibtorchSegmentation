#pragma once
#include<opencv2/opencv.hpp>

//contains mask and source image
struct Data {
	Data(cv::Mat img, cv::Mat _mask) :image(img), mask(_mask) {};
	cv::Mat image;
	cv::Mat mask;
};

class Augmentations
{
public:
	static Data Resize(Data mData, int width, int height, float probability);
	static Data HorizontalFlip(Data mData, float probability);
	static Data VerticalFlip(Data mData, float probability);
	static Data RandomScaleRotate(Data mData, float probability, float rotate_limit, \
								  float scale_limit, int interpolation, int boder_mode);
};


