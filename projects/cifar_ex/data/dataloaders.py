from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

import os


def get_loader(batch_size, num_workers, split: str):
    if split == 'train':
        train = True
    elif split == 'test':
        train = False
    else:
        raise ValueError

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    data_dir = '~/data/datasets'
    data_dir_expanded = os.path.expanduser(data_dir)
    dataset = CIFAR10(root=data_dir_expanded,
                      train=train,
                      download=True,
                      transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)

    return dataloader


def get_train_loader(batch_size: int = 1, num_workers: int = 1):
    trainloader = get_loader(batch_size, num_workers, 'train')
    return trainloader


def get_test_loader(batch_size: int = 1, num_workers: int = 1):
    testloader = get_loader(batch_size, num_workers, 'test')
    return testloader
