English | [中文](https://github.com/AllentDan/SegmentationCpp/blob/main/README-Chinese.md)

<div align="center">

![logo](https://raw.githubusercontent.com/AllentDan/ImageBase/main/OpenSource/LibtorchSegment.png)  
**C++ library with Neural Networks for Image  
Segmentation based on [LibTorch](https://pytorch.org/).**  

</div>

**⭐Please give a star if this project helps you.⭐**

The main features of this library are:

 - High level API (just a line to create a neural network)
 - 7 models architectures for binary and multi class segmentation (including legendary Unet)
 - 15 available encoders
 - All encoders have pre-trained weights for faster and better convergence
 - 35% or more inference speed boost compared with pytorch cuda, same speed for cpu. (Unet tested in rtx 2070s).
 
### [📚 Libtorch Tutorials 📚](https://github.com/AllentDan/LibtorchTutorials)

Visit [Libtorch Tutorials Project](https://github.com/AllentDan/LibtorchTutorials) if you want to know more about Libtorch Segment library.

### 📋 Table of content
 1. [Quick start](#start)
 2. [Examples](#examples)
 3. [Train your own data](#trainingOwn)
 4. [Models](#models)
    1. [Architectures](#architectures)
    2. [Encoders](#encoders)
 5. [Installation](#installation)
 6. [Thanks](#thanks)
 7. [To do list](#todo)
 8. [Citing](#citing)
 9. [License](#license)
 10. [Related repository](#related_repos)

### ⏳ Quick start <a name="start"></a>

#### 1. Create your first Segmentation model with Libtorch Segment

A resnet34 trochscript file is provided [here](https://github.com/AllentDan/LibtorchSegmentation/releases/download/weights/resnet34.pt). Segmentation model is just a LibTorch torch::nn::Module, which can be created as easy as:

```cpp
#include "Segmentor.h"
auto model = UNet(1, /*num of classes*/
                  "resnet34", /*encoder name, could be resnet50 or others*/
                  "path to resnet34.pt"/*weight path pretrained on ImageNet, it is produced by torchscript*/
                  );
```
 - see [table](#architectures) with available model architectures
 - see [table](#encoders) with available encoders and their corresponding weights

#### 2. Generate your own pretrained weights

All encoders have pretrained weights. Preparing your data the same way as during weights pre-training may give your better results (higher metric score and faster convergence). And you can also train only the decoder and segmentation head while freeze the backbone.

```python
import torch
from torchvision import models

# resnet34 for example
model = models.resnet34(pretrained=True)
model.eval()
var=torch.ones((1,3,224,224))
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("resnet34.pt")
```

Congratulations! You are done! Now you can train your model with your favorite backbone and segmentation framework.

### 💡 Examples <a name="examples"></a>
 - Training model for person segmentation using images from PASCAL VOC Dataset. "voc_person_seg" dir contains 32 json labels and their corresponding jpeg images for training and 8 json labels with corresponding images for validation.
```cpp
Segmentor<FPN> segmentor;
segmentor.Initialize(0/*gpu id, -1 for cpu*/,
                    512/*resize width*/,
                    512/*resize height*/,
                    {"background","person"}/*class name dict, background included*/,
                    "resnet34"/*backbone name*/,
                    "your path to resnet34.pt");
segmentor.Train(0.0003/*initial leaning rate*/,
                300/*training epochs*/,
                4/*batch size*/,
                "your path to voc_person_seg",
                ".jpg"/*image type*/,
                "your path to save segmentor.pt");
```

- Predicting test. A segmentor.pt file is provided in the project [here](https://github.com/AllentDan/LibtorchSegmentation/releases/download/weights/segmentor.pt). It is trained through a FPN with ResNet34 backbone for a few epochs. You can directly test the segmentation result through:
```cpp
cv::Mat image = cv::imread("your path to voc_person_seg\\val\\2007_004000.jpg");
Segmentor<FPN> segmentor;
segmentor.Initialize(0,512,512,{"background","person"},
                      "resnet34","your path to resnet34.pt");
segmentor.LoadWeight("segmentor.pt"/*the saved .pt path*/);
segmentor.Predict(image,"person"/*class name for showing*/);
```
the predicted result shows as follow:

![](https://raw.githubusercontent.com/AllentDan/SegmentationCpp/main/prediction.jpg)

### 🧑‍🚀 Train your own data <a name="trainingOwn"></a>
- Create your own dataset. Using [labelme](https://github.com/wkentaro/labelme) through "pip install" and label your images. Split the output json files and images into folders just like below:
```
Dataset
├── train
│   ├── xxx.json
│   ├── xxx.jpg
│   └......
├── val
│   ├── xxxx.json
│   ├── xxxx.jpg
│   └......
```
- Training or testing. Just like the example of "voc_person_seg", replace "voc_person_seg" with your own dataset path.
- Refer to [training tricks](https://github.com/AllentDan/LibtorchSegmentation/blob/main/docs/training%20tricks.md) to improve your final training performance.


### 📦 Models <a name="models"></a>

#### Architectures <a name="architectures"></a>
 - [x] Unet [[paper](https://arxiv.org/abs/1505.04597)]
 - [x] FPN [[paper](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)]
 - [x] PAN [[paper](https://arxiv.org/abs/1805.10180)]
 - [x] PSPNet [[paper](https://arxiv.org/abs/1612.01105)]
 - [x] LinkNet [[paper](https://arxiv.org/abs/1707.03718)]
 - [x] DeepLabV3 [[paper](https://arxiv.org/abs/1706.05587)]
 - [x] DeepLabV3+ [[paper](https://arxiv.org/abs/1802.02611)]


#### Encoders <a name="encoders"></a>
- [x] ResNet
- [x] ResNext
- [x] VGG

The following is a list of supported encoders in the Libtorch Segment. All the encoders weights can be generated through torchvision except resnest. Select the appropriate family of encoders and click to expand the table and select a specific encoder and its pre-trained weights.

<details>
<summary style="margin-left: 25px;">ResNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnet18                        |imagenet                        |11M                             |
|resnet34                        |imagenet                        |21M                             |
|resnet50                        |imagenet                        |23M                             |
|resnet101                       |imagenet                        |42M                             |
|resnet152                       |imagenet                        |58M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">ResNeXt</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnext50_32x4d                 |imagenet                        |22M                             |
|resnext101_32x8d                |imagenet                        |86M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">ResNeSt</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-resnest14d                 |imagenet                        |8M                              |
|timm-resnest26d                 |imagenet                        |15M                             |
|timm-resnest50d                 |imagenet                        |25M                             |
|timm-resnest101e                |imagenet                        |46M                             |
|timm-resnest200e                |imagenet                        |68M                             |
|timm-resnest269e                |imagenet                        |108M                            |
|timm-resnest50d_4s2x40d         |imagenet                        |28M                             |
|timm-resnest50d_1s4x24d         |imagenet                        |23M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SE-Net</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|senet154                        |imagenet                        |113M                            |
|se_resnet50                     |imagenet                        |26M                             |
|se_resnet101                    |imagenet                        |47M                             |
|se_resnet152                    |imagenet                        |64M                             |
|se_resnext50_32x4d              |imagenet                        |25M                             |
|se_resnext101_32x4d             |imagenet                        |46M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">VGG</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|vgg11                           |imagenet                        |9M                              |
|vgg11_bn                        |imagenet                        |9M                              |
|vgg13                           |imagenet                        |9M                              |
|vgg13_bn                        |imagenet                        |9M                              |
|vgg16                           |imagenet                        |14M                             |
|vgg16_bn                        |imagenet                        |14M                             |
|vgg19                           |imagenet                        |20M                             |
|vgg19_bn                        |imagenet                        |20M                             |

</div>
</details>

### 🛠 Installation <a name="installation"></a>
**Dependency:**

- [Opencv 3+](https://opencv.org/releases/)
- [Libtorch 1.7+](https://pytorch.org/)

**Windows:**

Configure the environment for libtorch development. [Visual studio](https://allentdan.github.io/2020/03/05/windows-libtorch-configuration/) and [Qt Creator](https://allentdan.github.io/2020/03/05/QT-Creator-Opencv-Libtorch-CUDA-English/) are verified for libtorch1.7x release.

**Linux && MacOS:**

Install libtorch and opencv. 

For libtorch, follow the official pytorch c++ tutorials [here](https://pytorch.org/tutorials/advanced/cpp_export.html).

For opencv, follow the official opencv install steps [here](https://github.com/opencv/opencv).

If you have already configured them both, congratulations!!! Download the pretrained weight [here](https://github.com/AllentDan/LibtorchSegmentation/releases/download/weights/resnet34.pt) and a demo .pt file [here](https://github.com/AllentDan/LibtorchSegmentation/releases/download/weights/segmentor.pt) into weights. 

Change the CMAKE_PREFIX_PATH to your own in CMakeLists.txt. Change the image path, pretrained path and segmentor path to your own in src/main.cpp. Then just in build folder, open the terminal, do the following:
```
cd build
cmake ..
make
./LibtorchSegmentation
```


### ⏳ ToDo <a name="todo"></a>
- [ ] More segmentation architectures and backbones
  - [ ] UNet++ [[paper](https://arxiv.org/pdf/1807.10165.pdf)]
  - [ ] ResNest
  - [ ] Se-Net
  - [ ] ...
- [x] Data augmentations
  - [x] Random horizontal flip
  - [x] Random vertical flip
  - [x] Random scale rotation
  - [ ] ...
- [x] Training tricks
  - [x] Combined dice and cross entropy loss
  - [x] Freeze backbone
  - [x] Multi step learning rate schedule
  - [ ] ...


### 🤝 Thanks <a name="thanks"></a>
By now, these projects helps a lot.
- [official pytorch](https://github.com/pytorch/pytorch)
- [qubvel SMP](https://github.com/qubvel/segmentation_models.pytorch)
- [wkentaro labelme](https://github.com/wkentaro/labelme)
- [nlohmann json](https://github.com/nlohmann/json)

### 📝 Citing
```
@misc{Chunyu:2021,
  Author = {Chunyu Dong},
  Title = {Libtorch Segment},
  Year = {2021},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/AllentDan/SegmentationCpp}}
}
```

### 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE). 

## Related repository <a name="related_repos"></a>
Based on libtorch, I released following repositories:
- [LibtorchTutorials](https://github.com/AllentDan/LibtorchTutorials)
- [LibtorchSegmentation](https://github.com/AllentDan/LibtorchSegmentation)
- [LibtorchDetection](https://github.com/AllentDan/LibtorchDetection)

Last but not least, **don't forget** your star...

Feel free to commit issues or pull requests, contributors wanted.

![stargazers over time](https://starchart.cc/AllentDan/SegmentationCpp.svg)
