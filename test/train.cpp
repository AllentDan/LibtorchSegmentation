#include <iostream>
#include <Segmentor.h>

int main(int argc, char *argv[])
{
    Segmentor<FPN> segmentor;
    segmentor.Initialize(-1, 512, 512, {"background","person"}, "resnet34", "./weights/resnet34.pt");
    segmentor.Train(0.0003, 300, 4, "./voc_person_seg", ".jpg", "./weights/segmentor.pt");

    return 0;
}
