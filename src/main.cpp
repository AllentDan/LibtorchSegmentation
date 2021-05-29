#include<iostream>
#include"Segmentor.h"

int main(int argc, char *argv[])
{
	//auto model = UNet(1, "resnet34", "D:\\AllentFiles\\code\\tmp\\resnet34.pt");
	//model->to(at::kCUDA);
	//model->eval();
	//auto input = torch::rand({ 1,3,512,512 }).to(at::kCUDA);
	//auto output = model->forward(input);
	//int T = 100;
	//int64 t0 = cv::getCPUTickCount();
	//for (int i = 0; i < T; i++) {
	//	auto output = model->forward(input);
	//	//output = output.to(at::kCPU);
	//}
	//output = output.to(at::kCPU);
	//int64 t1 = cv::getCPUTickCount();
	//std::cout << "execution time is " << (t1 - t0) / (double)cv::getTickFrequency() << " seconds" << std::endl;

    cv::Mat image = cv::imread("./voc_person_seg/val/2007_004000.jpg");

    Segmentor<FPN> segmentor;
    segmentor.Initialize(-1,512,512,{"background","person"},
                         "resnet34","./weights/resnet34.pt");
    segmentor.LoadWeight("./weights/segmentor.pt");
    segmentor.Predict(image,"person");

	//trainTricks tricks;

	////tricks for data augmentations
	//tricks.horizontal_flip_prob = 0.5;
	//tricks.vertical_flip_prob = 0.5;
	//tricks.scale_rotate_prob = 0.3;

	////tricks for training process
	//tricks.decay_epochs = { 40, 80 };
	//tricks.freeze_epochs = 8;

	//segmentor.SetTrainTricks(tricks);
    //segmentor.Train(0.0003,300,4,"D:\\AllentFiles\\data\\dataset4teach\\voc_person_seg",".jpg","segmentor.pt");

    return 0;
}
