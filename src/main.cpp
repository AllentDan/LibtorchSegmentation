#include<iostream>
#include"UNet.h"
#include"Segmentor.h"
#include"DeepLab.h"

int main(int argc, char *argv[])
{

	//torch::Device device = torch::Device(torch::kCUDA, 0);
	//auto model = DeepLabV3(1, "resnet34", "D:\\AllentFiles\\code\\tmp\\resnet34.pt");
	//model->eval();
	//model->to(device);
	//auto input = torch::rand({ 1,3,224,224 }).to(device);
	//model->forward(input);
	//int64 sstart = cv::getTickCount();
	//for (int i = 0; i < 10; i++) {
	//	model->forward(input);
	//}
	//double dduration = (cv::getTickCount() - sstart) / cv::getTickFrequency();
	//std::cout <<"Cost "<< dduration/10<<"s" << std::endl;
    cv::Mat image = cv::imread("D:\\AllentFiles\\data\\dataset4teach\\voc_person_seg\\val\\2007_004000.jpg");

    Segmentor<DeepLabV3Plus> segmentor;
    segmentor.Initialize(0,512,512,{"background","person"},
                         "resnext50_32x4d","D:\\AllentFiles\\code\\tmp\\resnext50_32x4d.pt");
    segmentor.LoadWeight("segmentor.pt");
    segmentor.Predict(image,"person");
    segmentor.Train(0.0003,300,2,"D:\\AllentFiles\\data\\dataset4teach\\voc_person_seg",".jpg","segmentor.pt");

    return 0;
}
