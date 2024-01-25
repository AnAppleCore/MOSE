import torch.nn as nn

from .my_transform import *


def get_transform(transform_name, input_size):
    if transform_name == 'ocm':
        transform = nn.Sequential(
            RandomColorGrayLayer(p=0.25),
            RandomResizedCropLayer(scale=(0.3, 1.0), size=[input_size[1], input_size[2], input_size[0]])
        )
    elif transform_name == 'simclr':
        transform = nn.Sequential(
            HorizontalFlipLayer(),
            RandomColorGrayLayer(p=0.25),
            RandomResizedCropLayer(scale=(0.3, 1.0), size=[input_size[1], input_size[2], input_size[0]])
        )
    elif transform_name == 'scr':
        transform = nn.Sequential(
            RandomResizedCropLayer(scale=(0.2, 1.0), size=[input_size[1], input_size[2], input_size[0]]),
            HorizontalFlipLayer(),
            ColorJitterLayer(0.8, 0.4, 0.4, 0.4, 0.1),
            RandomColorGrayLayer(p=0.2)
        )
    else:
        return None

    transform = transform.cuda()
    return transform