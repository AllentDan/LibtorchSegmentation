#include <iostream>
#include <Segmentor.h>

int main(int argc, char *argv[])
{

    cv::Mat image = cv::imread("./voc_person_seg/val/2007_003747.jpg");

    Segmentor<FPN> segmentor;
    segmentor.Initialize(-1,512,512,{"background","person"}, "resnet34","./weights/resnet34.pt");
    segmentor.LoadWeight("./weights/segmentor.pt");
    segmentor.Predict(image,"person");

    return 0;
}
