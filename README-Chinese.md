[English](https://github.com/AllentDan/SegmentationCpp) | 中文

<div align="center">

![logo](https://raw.githubusercontent.com/AllentDan/ImageBase/main/OpenSource/LibtorchSegment.png)  
**基于[LibTorch](https://pytorch.org/)的C++开源图像分割神经网络库.**  

</div>
**⭐如果有用请给我一个star⭐**

这个库具有以下优点:

 - 高级的API (只需一行代码就可创建网络)
 - 7 种模型架构可用于单类或者多类的分割任务 (包括Unet)
 - 7 种编码器网络
 - 所有的编码器都有预训练权重，可以更快更好地收敛
 - 相比于python下的GPU前向推理速度具有2倍或以上的优势, cpu下保持速度一致. (Unet测试于GTX 2070S).
 
### [📚 Libtorch教程 📚](https://github.com/AllentDan/LibtorchTutorials)

如果你想对该开源项目有更多更详细的了解，请前往本人另一个开源项目：[Libtorch教程](https://github.com/AllentDan/LibtorchTutorials) .

### 📋 目录
 1. [快速开始](#start)
 2. [例子](#examples)
 3. [训练自己的数据](#trainingOwn)
 4. [模型](#models)
    1. [架构](#architectures)
    2. [编码器](#encoders)
 5. [安装](#installation)
 6. [感谢](#thanks)
 7. [引用](#citing)
 8. [证书](#license)

### ⏳ 快速开始 <a name="start"></a>

#### 1. 用 Libtorch Segment 创建你的第一个分割网络

分割模型是 LibTorch 的 torch::nn::Module的派生类, 可以很容易生成:

```cpp
#include "Segmentor.h"
auto model = UNet(1, /*num of classes*/
                  "resnet34", /*encoder name, could be resnet50 or others*/
                  "path to resnet34.pt"/*weight path pretrained on ImageNet, it is produced by torchscript*/
                  );
```
 - 见 [表](#architectures) 查看所有支持的模型架构
 - 见 [表](#encoders) 查看所有的编码器网络和相应的预训练权重

#### 2. 生成自己的预训练权重

所有编码器均具有预训练的权重。加载预训练权重，以相同的方式训练数据，可能会获得更好的结果（更高的指标得分和更快的收敛速度）。还可以在冻结主干的同时仅训练解码器和分割头。

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

恭喜你！ 大功告成！ 现在，您可以使用自己喜欢的主干和分割框架来训练模型了。

### 💡 例子 <a name="examples"></a>
 - 使用来自PASCAL VOC数据集的图像进行人体分割数据训练模型. "voc_person_seg" 目录包含32个json标签及其相应的jpeg图像用于训练，还有8个json标签以及相应的图像用于验证。
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

- 预测测试。项目中提供了以ResNet34为骨干网络的FPN网络，训练了一些周期得到segmentor.pt文件。 您可以直接测试分割结果:
```cpp
cv::Mat image = cv::imread("your path to voc_person_seg\\val\\2007_004000.jpg");
Segmentor<FPN> segmentor;
segmentor.Initialize(0,512,512,{"background","person"},
                      "resnet34","your path to resnet34.pt");
segmentor.LoadWeight("segmentor.pt"/*the saved .pt path*/);
segmentor.Predict(image,"person"/*class name for showing*/);
```
预测结果显示如下:

![](https://raw.githubusercontent.com/AllentDan/SegmentationCpp/main/prediction.jpg)

### 🧑‍🚀 训练自己的数据 <a name="trainingOwn"></a>
- 创建自己的数据集. 使用"pip install"安装[labelme](https://github.com/wkentaro/labelme)并标注你的图像. 将输出的json文件和图像分成以下文件夹：
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
- 训练或测试。就像“ voc_person_seg”的示例一样，用自己的数据集路径替换“ voc_person_seg”。


### 📦 Models <a name="models"></a>

#### Architectures <a name="architectures"></a>
 - [x] Unet [[paper](https://arxiv.org/abs/1505.04597)]
 - [x] FPN [[paper](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)]
 - [x] PAN [[paper](https://arxiv.org/abs/1805.10180)]
 - [x] PSPNet [[paper](https://arxiv.org/abs/1612.01105)]
 - [x] LinkNet [[paper](https://arxiv.org/abs/1707.03718)]
 - [x] DeepLabV3 [[paper](https://arxiv.org/abs/1706.05587)]
 - [x] DeepLabV3+ [[paper](https://arxiv.org/abs/1802.02611)]
 - [ ] UNet++ [[paper](https://arxiv.org/pdf/1807.10165.pdf)]

#### Encoders <a name="encoders"></a>
- [x] ResNet
- [x] ResNext
- [ ] ResNest
- [ ] Se-Net

以下是该项目中受支持的编码器的列表。除resnest外，所有编码器权重都可以通过torchvision生成。选择适当的编码器，然后单击以展开表格，然后选择特定的编码器及其预训练的权重。

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

### 🛠 安装 <a name="installation"></a>
**依赖库:**

- [Opencv 3+](https://opencv.org/releases/)
- [Libtorch 1.7+](https://pytorch.org/)

**Windows:**

配置libtorch 开发环境. [Visual studio](https://allentdan.github.io/2020/12/16/pytorch%E9%83%A8%E7%BD%B2torchscript%E7%AF%87) 和 [Qt Creator](https://allentdan.github.io/2021/01/21/QT%20Creator%20+%20Opencv4.x%20+%20Libtorch1.7%E9%85%8D%E7%BD%AE/#more)已经通过libtorch1.7x release的验证. 

**Linux && MacOS:**

按照官方提供的pytorch c++ [部署](https://pytorch.org/tutorials/advanced/cpp_export.html). 比Windows要简单许多.

### 🤝 感谢 <a name="thanks"></a>
这个项目还在施工，以下是目前给予帮助的项目.
- [official pytorch](https://github.com/pytorch/pytorch)
- [qubvel SMP](https://github.com/qubvel/segmentation_models.pytorch)
- [wkentaro labelme](https://github.com/wkentaro/labelme)
- [nlohmann json](https://github.com/nlohmann/json)

### 📝 引用
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

### 🛡️ 证书 <a name="license"></a>
该项目以 [MIT License](https://github.com/qubvel/segmentation_models.pytorch/blob/master/LICENSE)开源，别忘了点赞哟
![stargazers over time](https://starchart.cc/AllentDan/SegmentationCpp.svg)
