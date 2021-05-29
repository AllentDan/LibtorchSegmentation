#include "Augmentations.h"

template<typename T>
T RandomNum(T _min, T _max)
{
	T temp;
	if (_min > _max)
	{
		temp = _max;
		_max = _min;
		_min = temp;
	}
	return rand() / (double)RAND_MAX *(_max - _min) + _min;
}


cv::Mat centerCrop(cv::Mat srcImage, int width, int height) {
	int srcHeight = srcImage.rows;
	int srcWidth = srcImage.cols;
	int maxHeight = srcHeight > height ? srcHeight : height;
	int maxWidth = srcWidth > width ? srcWidth : width;
	cv::Mat maxImage = cv::Mat::zeros(cv::Size(maxWidth, maxHeight), srcImage.type());
	int h_max_start = int((maxHeight - srcHeight) / 2);
	int w_max_start = int((maxWidth - srcWidth) / 2);
	srcImage.clone().copyTo(maxImage(cv::Rect(w_max_start, h_max_start, srcWidth, srcHeight)));

	int h_start = int((maxHeight - height) / 2);
	int w_start = int((maxWidth - width) / 2);
	cv::Mat dstImage = maxImage(cv::Rect(w_start, h_start, width, height)).clone();
	return dstImage;
}

cv::Mat RotateImage(cv::Mat src, float angle, float scale, int interpolation, int boder_mode)
{
	cv::Mat dst;

	//make output size same with input after scaling
	cv::Size dst_sz(src.cols, src.rows);
	scale = 1 + scale;
	cv::resize(src, src, cv::Size(int(src.cols*scale), int(src.rows*scale)));
	src = centerCrop(src, dst_sz.width, dst_sz.height);

	//center for rotating 
	cv::Point2f center(static_cast<float>(src.cols / 2.), static_cast<float>(src.rows / 2.));

	//rotate matrix     
	cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

	cv::warpAffine(src, dst, rot_mat, dst_sz, interpolation, boder_mode);
	return dst;
}


Data Augmentations::Resize(Data mData, int width, int height, float probability) {
	float rand_number = RandomNum<float>(0, 1);
	if (rand_number <= probability) {

		float h_scale = height * 1.0 / mData.image.rows;
		float w_scale = width * 1.0 / mData.image.cols;

		cv::resize(mData.image, mData.image, cv::Size(width, height));
		cv::resize(mData.mask, mData.mask, cv::Size(width, height));
	}
	return mData;
}

Data Augmentations::HorizontalFlip(Data mData, float probability) {
	float rand_number = RandomNum<float>(0, 1);
	if (rand_number <= probability) {

		cv::flip(mData.image, mData.image, 1);
		cv::flip(mData.mask, mData.mask, 1);

	}
	return mData;
}

Data Augmentations::VerticalFlip(Data mData, float probability) {
	float rand_number = RandomNum<float>(0, 1);
	if (rand_number <= probability) {

		cv::flip(mData.image, mData.image, 0);
		cv::flip(mData.mask, mData.mask, 0);

	}
	return mData;
}

Data Augmentations::RandomScaleRotate(Data mData, float probability, float rotate_limit, float scale_limit, int interpolation, int boder_mode) {
	float rand_number = RandomNum<float>(0, 1);
	if (rand_number <= probability) {
		float angle = RandomNum<float>(-rotate_limit, rotate_limit);
		float scale = RandomNum<float>(-scale_limit, scale_limit);
		mData.image = RotateImage(mData.image, angle, scale, interpolation, boder_mode);
		mData.mask = RotateImage(mData.mask, angle, scale, interpolation, boder_mode);
		return mData;
	}
	return mData;
}