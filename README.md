# torch_mask_rcnn
It is a repository from mask_rcnn using pytorch 1.4.0 with CUDA

## Setup Instructions
Assume pytorch 1.4.0 with CUDA and ROS Melodic are installed, please run the following instructions:

```
$ pip3 install torchvision==0.5.0
$ sudo apt-get install python3-matplotlib
```

After the above installation, please download the network from the link below and copy to */home/vincent/.cache/torch/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth*

https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth

## Usages of python scripts
1) torch_mask_rcnn.py : It is the original python script which will output the segmentation of car.jpg with plt. This script is the cornerstone of extensive development.

2) torch_mask_rcnn_cv.py: It is the script based on (1) and replace plt with opencv.



