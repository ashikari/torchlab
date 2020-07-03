from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        block_elems = [nn.Conv2d(in_channels=channels,
                                 out_channels=channels,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=padding),
                       nn.ReLU(True),
                       nn.BatchNorm2d(channels),
                       nn.Conv2d(in_channels=channels,
                                 out_channels=channels,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=padding),
                       nn.BatchNorm2d(channels)]

        self.block = nn.Sequential(*block_elems)

    def forward(self, x):

        return F.relu(self.block(x) + x)


class ResNet(nn.Module):

    def __init__(self, channel_list, kernel_size=3, stride=2):
        super().__init__()
        self.channel_list = channel_list
        self.stride = stride
        self.network = nn.ModuleList()
        padding = (kernel_size - 1) // 2

        for idx in range(len(channel_list) - 1):
            net_def_list = [ResBlock(channel_list[idx]),
                            nn.Conv2d(in_channels=channel_list[idx],
                                      out_channels=channel_list[idx + 1],
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding),
                            nn.ReLU(True),
                            nn.BatchNorm2d(channel_list[idx + 1])]
            self.network.append(nn.Sequential(*net_def_list))

    def forward(self, x):
        feat_maps = []
        feat_maps.append(self.network[0](x))

        for i in range(1, len(self.network)):
            feat_maps.append(self.network[i](feat_maps[-1]))

        return feat_maps

