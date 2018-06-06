# MUNet
MUNet: Macro Unit-based Convolutional Neural Networks for Mobile Devices

#### Dependencies

 * Python 3.5+
 * Tensorflow 1.2.0+
 * Keras 2.0.6+
 
#### Usage
 1. Cloning the repository

```
$ git clone https://github.com/kdhht2334/MUNet-CVPR18-Workshop.git
$ cd MUNET-CVPR18-Workshop/
```

 2. Run

```
$ python main.py --gpus gpu-id
```
For example, gpu-id is from 0 to 3 in case of 4GPUs device.

If you use 1GPU device, then gpu-id is 0.

 
## Milestone
  - [x] Cifar dataset experiments
  - [ ] SVHN dataset experiments
  - [ ] Tiny ImageNet dataset experiments
  - [ ] Add results table
 
## Citation

If this work is helpful for your research, please cite our:
```
 @InProceedings{Kim_2018_CVPR_Workshops,
author = {Ha Kim, Dae and Hyun Lee, Seung and Cheol Song, Byung},
title = {MUNet: Macro Unit-Based Convolutional Neural Network for Mobile Devices},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
}
```