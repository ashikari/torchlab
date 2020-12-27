
"""
Define the network model
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from infra.backbone import ResNet

class CifarResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = [3, 64, 128, 128, 256]
        self.backbone = ResNet(channel_list=self.channels, stride=1)
        self.fully_connected = nn.Sequential(nn.Linear(in_features=self.channels[-1] * 32 * 32,
                                                       out_features=256),
                                             nn.ReLU(True),
                                             nn.Linear(in_features=256,
                                                       out_features=10))


    def forward(self, x):
        """
        Perform network forward pass
        """
        feat_maps = self.backbone(x)
        flat_feat = feat_maps[-1].view(-1, self.channels[-1] * 32 * 32)
        return self.fully_connected(flat_feat)

    def predict(self, x):
        x = self.forward(x)
        return F.softmax(x)

    def get_state_dict_all(self):
        state_dict = {"backbone": self.backbone.state_dict(),
                      "fully_connected": self.fully_connected.state_dict()}
        return state_dict

    def load_state_dict(self, state_dict):
        print("Loading Checkpoint")
        if "backbone" in state_dict.keys():
            self.backbone.load_state_dict(state_dict["backbone"])
        if "fully_connected" in state_dict.keys():
            self.fully_connected.load_state_dict(state_dict["fully_connected"])

class CifarNet(nn.Module):

    def __init__(self):
        super().__init__()

        kernel_size = 3
        padding = (kernel_size - 1) // 2

        # define convolutions
        channels = [3, 50, 100, 200]

        convs = []

        for idx, channel_num in enumerate(channels[:-2]):
            input_channels = channel_num
            output_channels = channels[idx + 1]

            block = [
                nn.Conv2d(in_channels=input_channels,
                          out_channels=output_channels,
                          kernel_size=kernel_size,
                          stride=1,
                          padding=padding),
                nn.ReLU(True),
                nn.BatchNorm2d(output_channels)
            ]
            convs += block

        # last conv is a 1x1 to reduce the number of required parameters
        convs.append(nn.Conv2d(in_channels=channels[-2],
                               out_channels=channels[-1],
                               kernel_size=1,
                               stride=1))
        self.convs = nn.Sequential(*convs)

        # define fully connected layers
        self.fully_connected = nn.Sequential(nn.Linear(in_features=channels[-1] * 32 * 32,
                                                       out_features=200),
                                             nn.ReLU(True),
                                             nn.Linear(in_features=200,
                                                       out_features=10))

    def forward(self, x):
        """
        Perform network forward pass
        """
        x = self.convs(x)
        x = x.view(-1, 200*32*32)
        return self.fully_connected(x)

    def get_state_dict_all(self):
        return self.state_dict()

    def load_state_dict_all(self, state_dict):
        self.load_state_dict(state_dict)
