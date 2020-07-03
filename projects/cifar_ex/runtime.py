"""
Contains the labs that will train the network
"""
import numpy
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam

from infra.lab import Lab
from projects.cifar_ex.nets import CifarNet, CifarResNet
from projects.cifar_ex.data.dataloaders import get_train_loader


class CifarLab(Lab):

    label_map = ('plane',
                 'car',
                 'bird',
                 'cat',
                 'deer',
                 'dog',
                 'frog',
                 'horse',
                 'ship',
                 'truck')

    def __init__(self, exp_name: str, batch_size: int = 1, n_epochs: int = 10):

        # get model
        model = self.get_model()
        if torch.cuda.is_available():
            model.cuda()

        # get train dl
        train_dl = self.get_train_dl(batch_size=batch_size)

        # get optimizer
        optimizer = self.get_optimizer(model.parameters())

        super().__init__(exp_name, model, train_dl,
                         optimizer, log_interval=200, n_epochs=n_epochs)

        self.classification_loss = nn.CrossEntropyLoss()

    # TODO: restructure code to make the forward pass happen in the loss fucntion
    def loss_function(self, dataloader_output):
        image, gt_label = dataloader_output
        if torch.cuda.is_available():
            # input image
            image = image.cuda()
            # Ground Truth
            gt_label = gt_label.cuda()
        # model estimated label
        est_label = self.model(image)
        return self.classification_loss(est_label, gt_label)

    @staticmethod
    def get_optimizer(params, lr=3e-4):
        return Adam(params=params, lr=lr)

    @staticmethod
    def get_model():
        # return CifarNet()
        return CifarResNet()

    @staticmethod
    def get_train_dl(batch_size: int = 1):
        return get_train_loader(batch_size)
