<div align="center">
 
![logo](https://raw.githubusercontent.com/AllentDan/ImageBase/main/OpenSource/LibtorchSegment.png)  
**C++ library with Neural Networks for Image  
Segmentation based on [LibTorch](https://pytorch.org/).**  

</div>

The main features of this library are:

 - High level API (just a line to create a neural network)
 - 2 models architectures for binary and multi class segmentation (including legendary Unet)
 - 5 available encoders
 - All encoders have pre-trained weights for faster and better convergence
 
### [📚 Libtorch Tutorials 📚](https://github.com/AllentDan/LibtorchTutorials/tree/master)

Visit [Libtorch Tutorials Project](https://github.com/AllentDan/LibtorchTutorials/tree/master) if you want to know more about Libtorch Segment library.

### 📋 Table of content
 1. [Quick start](#start)
 2. [Examples](#examples)
 3. [Models](#models)
    1. [Architectures](#architectures)
    2. [Encoders](#encoders)
 4. [Installation](#installation)
 5. [Thanks](#thanks)
 6. [Citing](#citing)
 7. [License](#license)

### ⏳ Quick start <a name="start"></a>

#### 1. Create your first Segmentation model with Libtorch Segment

Segmentation model is just a LibTorch torch::nn::Module, which can be created as easy as:

```cpp
#include "Segmentor.h"
auto model = UNet(1, //num of classes
                  "resnet34", //encoder name, could be resnet50 or others
                  "path to resnet34.pt"//weight path pretrained on ImageNet, it is produced by torchscript
                  );
```
 - see [table](#architectires) with available model architectures
 - see [table](#encoders) with available encoders and their corresponding weights

#### 2. Generate your own pretrained weights

All encoders have pretrained weights. Preparing your data the same way as during weights pre-training may give your better results (higher metric score and faster convergence). And you can also train only the decoder and segmentation head while freeze the backbone.

```python
import torch
from torchvision import models

# resnet50 for example
model = models.resnet50(pretrained=True)
model.eval()
var=torch.ones((1,3,224,224))
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("resnet50.pt")
```

Congratulations! You are done! Now you can train your model with your favorite backbone and segmentation framework.

### 💡 Examples <a name="examples"></a>
 - Training model for person segmentation on using images from PASCAL VOC Dataset. voc_person_seg dir contains 32 json labels and their corresponding jpeg images for training and 8 json labels with images for validation.
```cpp
Segmentor<FPN> segmentor;
segmentor.Initialize(0,512,512,{"background","person"},
                      "resnet34","your path to resnet34.pt");
segmentor.Train(0.0003,300,4,"your path to voc_person_seg",".jpg","your path to save segmentor.pt");
```

- Predicting test. A segmentor.pt file is provided in the project. You can directly test the segmentation result throgh:
```cpp
cv::Mat image = cv::imread("your path to voc_person_seg\\val\\2007_004000.jpg");
Segmentor<FPN> segmentor;
segmentor.Initialize(0,512,512,{"background","person"},
                      "resnet34","your path to resnet34.pt");
segmentor.LoadWeight("segmentor.pt");
segmentor.Predict(image,"person");
```
the predicted result shows as follow:

![](https://raw.githubusercontent.com/AllentDan/SegmentationCpp/main/prediction.jpg)

### 📦 Models <a name="models"></a>

#### Architectures <a name="architectures"></a>
 - [x] Unet [[paper](https://arxiv.org/abs/1505.04597)] [[docs](https://smp.readthedocs.io/en/latest/models.html#unet)] 
 - [x] FPN [[paper](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)] [[docs](https://smp.readthedocs.io/en/latest/models.html#fpn)]
 - [ ] PSPNet [[paper](https://arxiv.org/abs/1612.01105)] [[docs](https://smp.readthedocs.io/en/latest/models.html#pspnet)]
 - [ ] PAN [[paper](https://arxiv.org/abs/1805.10180)] [[docs](https://smp.readthedocs.io/en/latest/models.html#pan)]
 - [ ] DeepLabV3 [[paper](https://arxiv.org/abs/1706.05587)] [[docs](https://smp.readthedocs.io/en/latest/models.html#deeplabv3)]
 - [ ] DeepLabV3+ [[paper](https://arxiv.org/abs/1802.02611)] [[docs](https://smp.readthedocs.io/en/latest/models.html#id9)]

#### Encoders <a name="encoders"></a>
- [x] VGG
- [x] ResNet
- [ ] ResNext
- [ ] ResNest

The following is a list of supported encoders in the SMP. Select the appropriate family of encoders and click to expand the table and select a specific encoder and its pre-trained weights (`encoder_name` and `encoder_weights` parameters).

<details>
<summary style="margin-left: 25px;">ResNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnet18                        |imagenet / ssl / swsl           |11M                             |
|resnet34                        |imagenet                        |21M                             |
|resnet50                        |imagenet / ssl / swsl           |23M                             |
|resnet101                       |imagenet                        |42M                             |
|resnet152                       |imagenet                        |58M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">ResNeXt</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnext50_32x4d                 |imagenet / ssl / swsl           |22M                             |
|resnext101_32x4d                |ssl / swsl                      |42M                             |
|resnext101_32x8d                |imagenet / instagram / ssl / swsl|86M                         |
|resnext101_32x16d               |instagram / ssl / swsl          |191M                            |
|resnext101_32x32d               |instagram                       |466M                            |
|resnext101_32x48d               |instagram                       |826M                            |

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
Windows:
Confige the environment for libtorch development. Visual studio and Qt Creator are verified for libtorch1.7x release.

Linux && MacOS:
Follow the official pytorch c++ tutorials [here](https://pytorch.org/tutorials/advanced/cpp_export.html). It can be no more difficult than windows.

### 🤝 Thanks <a name="thanks"></a>
This project is under developing. By now, these projects helps a lot.
- [official pytorch](https://github.com/pytorch/pytorch)
- [qubvel SMP](https://github.com/qubvel/segmentation_models.pytorch)
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
Project is distributed under [MIT License](https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE)
