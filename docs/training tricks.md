## Training tricks
We provide a struct **trianTricks** for user to get better training performance. Users can apply the setting by:
```
trianTricks tricks;
segmentor.SetTrainTricks(tricks);
```

### Data augmentations
Through OpenCV, we can also make data augmentations during training procedure. This repository mainly provide the following data augmentations:
- Random horizontal flip
- Random vertical flip
- Random scale rotation
For horizontal and vertical flip, the probabilities to implement them are controlled by:
```
horizontal_flip_prob (float): probability to do horizontal flip augmentation, default 0;
vertical_flip_prob (float): probability to do vertical flip augmentation, default 0;
```
For random scale and rotate augmentation, it is set as:
```
scale_rotate_prob (float): probability to do rotate and scale augmentation, default 0;
```
Default 0 means this augmentation will be applied with 0 probability during the training procedure, not applied in other words.

Besides, we also provide scale and rotate limitation parameters, the interpolation method and the padding mode for this augmentation.
```
scale_limit (float): random enlarge or shrink the image by scale limit. For instance, if scale_limit equal to 0.1, \
                     it will random resize the image size to [size*(0.9), size*(1.1)];
rotate_limit (float): random rotate the image with the angle of [-rotate_limit, rotate_limit], in degree measure. \
                     45 degrees by default.
interpolation (int): It use opencv interpolation setting, cv::INTER_LINEAR by default.
border_mode (int): It use opencv border type setting, cv::BORDER_CONSTANT by default.
```

### Loss
For better training, we can also use different segmentation loss functions. This repository mainly provide the following loss functions:
- Cross entropy loss funcion
- Dice loss function
We can control the final loss by a hyper-parameter:
```
dice_ce_ratio (float): the weight of dice loss in combind loss, default 0.5;
```
The final loss used during training will be:
```
loss = DiceLoss * dice_ce_ratio +  CELoss * (1-dice_ce_ratio)
```

### Freeze backbone
Just like pytorch, we can also freeze the backbone of the segmentation network. This repository provide a parameter to contol this:
```
freeze_epochs (unsigned int): freeze the backbone during the first freeze_epochs, default 0;
```
By default, the training will not use it.

### Decay the learning rate
Well, libtorch does not provide learning rate schedure function for users. But we can also design it by ourselves. In this repository, we a multi step decay schedure.
```
decay_epochs (std::vector<unsigned int>): every decay_epoch, learning rate will decay by 90 percent, default {0};
```
By default, this schedure will not be used during training.

## Some advise or wishes
All the training tricks used in pytorch or python can be implemented in libtorch or cpp. But aparently, this reposity can not fullfill all of them. If you need a specific trick for the training, just implement it.

**Any pull request to share your tricks or networks will be welcomed.**