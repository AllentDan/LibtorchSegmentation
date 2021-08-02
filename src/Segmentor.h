/*
This libtorch implementation is writen by AllentDan.
Copyright(c) AllentDan 2021,
All rights reserved.
*/
#ifndef SEGMENTOR_H
#define SEGMENTOR_H
#include"architectures/FPN.h"
#include"architectures/PAN.h"
#include"architectures/UNet.h"
#include"architectures/LinkNet.h"
#include"architectures/PSPNet.h"
#include"architectures/DeepLab.h"
#include"utils/loss.h"
#include"SegDataset.h"
#include <sys/stat.h>
#if _WIN32
#include <io.h>
#else
#include<unistd.h>
#endif

template <class Model>
class Segmentor
{
public:
	Segmentor();
	~Segmentor() {};
	void Initialize(int gpu_id, int width, int height, std::vector<std::string> name_list,
		std::string encoder_name, std::string pretrained_path);
	void SetTrainTricks(trainTricks &tricks);
	void Train(float learning_rate, int epochs, int batch_size,
		std::string train_val_path, std::string image_type, std::string save_path);
	void LoadWeight(std::string weight_path);
	void Predict(cv::Mat image, std::string which_class);
private:
	int width = 512; int height = 512; std::vector<std::string> name_list;
	torch::Device device = torch::Device(torch::kCPU);
	trainTricks tricks;
	//    FPN fpn{nullptr};
	//    UNet unet{nullptr};
	Model model{ nullptr };
};

template <class Model>
Segmentor<Model>::Segmentor()
{
};

template <class Model>
void Segmentor<Model>::Initialize(int gpu_id, int _width, int _height, std::vector<std::string> _name_list,
	std::string encoder_name, std::string pretrained_path) {
	width = _width;
	height = _height;
	name_list = _name_list;
	//std::cout << pretrained_path << std::endl;
	//struct stat s {};
	//lstat(pretrained_path.c_str(), &s);
#ifdef _WIN32
	if ((_access(pretrained_path.data(), 0)) == -1)
	{
		std::cout<< "Pretrained path is invalid";
	}
#else
	if (access(pretrained_path.data(), F_OK) != 0)
	{
		std::cout<< "Pretrained path is invalid";
	}
#endif
	if (name_list.size() < 2) std::cout<<  "Class num is less than 1";
	int gpu_num = torch::getNumGPUs();
	if (gpu_id >= gpu_num) std::cout<< "GPU id exceeds max number of gpus";
	if (gpu_id >= 0) device = torch::Device(torch::kCUDA, gpu_id);

	model = Model(name_list.size(), encoder_name, pretrained_path);
	//    fpn = FPN(name_list.size(),encoder_name,pretrained_path);
	model->to(device);
}

template<class Model>
void Segmentor<Model>::SetTrainTricks(trainTricks &tricks) {
	this->tricks = tricks;
	return;
}

template <class Model>
void Segmentor<Model>::Train(float learning_rate, int epochs, int batch_size,
	std::string train_val_path, std::string image_type, std::string save_path) {

	std::string train_dir = train_val_path.append({ file_sepator() }).append("train");
	std::string val_dir = replace_all_distinct(train_dir,"train","val");

	std::vector<std::string> list_images_train = {};
	std::vector<std::string> list_labels_train = {};
	std::vector<std::string> list_images_val = {};
	std::vector<std::string> list_labels_val = {};

	load_seg_data_from_folder(train_dir, image_type, list_images_train, list_labels_train);
	load_seg_data_from_folder(val_dir, image_type, list_images_val, list_labels_val);

	auto custom_dataset_train = SegDataset(width, height, list_images_train, list_labels_train, \
										   name_list, tricks, true).map(torch::data::transforms::Stack<>());
	auto custom_dataset_val = SegDataset(width, height, list_images_val, list_labels_val, \
		                                 name_list, tricks, false).map(torch::data::transforms::Stack<>());
	auto options = torch::data::DataLoaderOptions();
	options.drop_last(true);
	options.batch_size(batch_size);
	auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_train), options);
	auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), options);

	for (int epoch = 0; epoch < epochs; epoch++) {
		float loss_sum = 0;
		int batch_count = 0;
		float loss_train = 0;
		float dice_coef_sum = 0;
		float best_loss = 1e10;

		for (auto decay_epoch : tricks.decay_epochs) {
			if(decay_epoch-1 == epoch)
				learning_rate /= 10;
		}
		torch::optim::Adam optimizer(model->parameters(), learning_rate);
		if (epoch < tricks.freeze_epochs) {
			for (auto mm : model->named_parameters())
			{
				if (strstr(mm.key().data(), "encoder"))
				{
					mm.value().set_requires_grad(false);
				}
				else
				{
					mm.value().set_requires_grad(true);
				}
			}
		}
		else {
			for (auto mm : model->named_parameters())
			{
				mm.value().set_requires_grad(true);
			}
		}
		model->train();
		for (auto& batch : *data_loader_train) {
			auto data = batch.data;
			auto target = batch.target;
			data = data.to(torch::kF32).to(device).div(255.0);
			target = target.to(torch::kLong).to(device).squeeze(1);//.clamp_max(1);//if you choose clamp, all classes will be set to only one

			optimizer.zero_grad();
			// Execute the model
			torch::Tensor prediction = model->forward(data);
			// Compute loss value
			torch::Tensor ce_loss = CELoss(prediction, target);
			torch::Tensor dice_loss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), name_list.size());
			auto loss = dice_loss * tricks.dice_ce_ratio + ce_loss * (1 - tricks.dice_ce_ratio);
			// Compute gradients
			loss.backward();
			// Update the parameters
			optimizer.step();
			loss_sum += loss.item().toFloat();
			dice_coef_sum += (1- dice_loss).item().toFloat();
			batch_count++;
			loss_train = loss_sum / batch_count / batch_size;
			auto dice_coef = dice_coef_sum / batch_count;

			std::cout << "Epoch: " << epoch << "," << " Training Loss: " << loss_train << \
											   "," << " Dice coefficient: " << dice_coef << "\r";
		}
		std::cout << std::endl;
		// validation part
		model->eval();
		loss_sum = 0; batch_count = 0; dice_coef_sum = 0;
		float loss_val = 0;
		for (auto& batch : *data_loader_val) {
			auto data = batch.data;
			auto target = batch.target;
			data = data.to(torch::kF32).to(device).div(255.0);
			target = target.to(torch::kLong).to(device).squeeze(1);//.clamp_max(1);

			// Execute the model
			torch::Tensor prediction = model->forward(data);

			// Compute loss value
			torch::Tensor ce_loss = CELoss(prediction, target);
			torch::Tensor dice_loss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), name_list.size());
			auto loss = dice_loss * tricks.dice_ce_ratio + ce_loss * (1 - tricks.dice_ce_ratio);
			loss_sum += loss.item<float>();
			dice_coef_sum += (1 - dice_loss).item().toFloat();
			batch_count++;
			loss_val = loss_sum / batch_count / batch_size;
			auto dice_coef = dice_coef_sum / batch_count;

			std::cout << "Epoch: " << epoch << "," << " Validation Loss: " << loss_val << \
											   "," << " Dice coefficient: " << dice_coef << "\r";
		}
		std::cout << std::endl;
		if (loss_val < best_loss) {
			torch::save(model, save_path);
			best_loss = loss_val;
		}
	}
	return;
}

template <class Model>
void Segmentor<Model>::LoadWeight(std::string weight_path) {
	torch::load(model, weight_path);
	model->to(device);
	model->eval();
	return;
}

template <class Model>
void Segmentor<Model>::Predict(cv::Mat image, std::string which_class) {
	cv::Mat srcImg = image.clone();
	int which_class_index = -1;
	for (int i = 0; i < name_list.size(); i++) {
		if (name_list[i] == which_class) {
			which_class_index = i;
			break;
		}
	}
	if (which_class_index == -1) std::cout<< which_class + "not in the name list";
	int image_width = image.cols;
	int image_height = image.rows;
	cv::resize(image, image, cv::Size(width, height));
	torch::Tensor tensor_image = torch::from_blob(image.data, { 1, height, width,3 }, torch::kByte);
	tensor_image = tensor_image.to(device);
	tensor_image = tensor_image.permute({ 0,3,1,2 });
	tensor_image = tensor_image.to(torch::kFloat);
	tensor_image = tensor_image.div(255.0);

	try
	{
		at::Tensor output = model->forward({ tensor_image });

	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
	}
	at::Tensor output = model->forward({ tensor_image });
	output = torch::softmax(output, 1).mul(255.0).toType(torch::kByte);

	image = cv::Mat::ones(cv::Size(width, height), CV_8UC1);

	at::Tensor re = output[0][which_class_index].to(at::kCPU).detach();
	memcpy(image.data, re.data_ptr(), width * height * sizeof(unsigned char));
	cv::resize(image, image, cv::Size(image_width, image_height));

	// draw the prediction
	cv::imwrite("prediction.jpg", image);
	cv::imshow("prediction", image);
	cv::imshow("srcImage", srcImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return;
}

#endif // SEGMENTOR_H
