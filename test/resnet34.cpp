#include <Segmentor.h>

#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr,
            "usage:\n resnet34 person_image backbone_path "
            "segmentor_path\nexample:\n./resnet34"
            " ../../voc_person_seg/val/2007_003747.jpg"
            " ../../weights/resnet34.pt ../../weights/segmentor.pt");
    return 1;
  }
  auto image_path = argv[1];
  auto backbone_path = argv[2];
  auto segmentor_path = argv[3];

  cv::Mat image = cv::imread(image_path);

  Segmentor<FPN> segmentor;
  segmentor.Initialize(0, 512, 512, {"background", "person"}, "resnet34",
                       backbone_path);
  segmentor.LoadWeight(segmentor_path);
  segmentor.Predict(image, "person");

  return 0;
}
