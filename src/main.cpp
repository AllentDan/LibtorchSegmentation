#include<iostream>
#include"UNet.h"
#include"Segmentor.h"

int main(int argc, char *argv[])
{

	//auto model = UNet(1, "resnext50_32x4d", "D:\\AllentFiles\\code\\tmp\\resnext50_32x4d.pt");
    cv::Mat image = cv::imread("D:\\AllentFiles\\data\\dataset4teach\\voc_person_seg\\val\\2007_004476.jpg");

    Segmentor<FPN> segmentor;
    segmentor.Initialize(0,512,512,{"background","person"},
                         "resnext50_32x4d","D:\\AllentFiles\\code\\tmp\\resnext50_32x4d.pt");
    segmentor.LoadWeight("segmentor.pt");
    segmentor.Predict(image,"person");
    segmentor.Train(0.0003,300,4,"D:\\AllentFiles\\data\\dataset4teach\\voc_person_seg",".jpg","segmentor.pt");

    return 0;
}
