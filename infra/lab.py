import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


class Lab:

    def __init__(self,
                 name: str,
                 model: nn.Module,
                 train_dl: DataLoader,
                 optimizer: Optimizer,
                 n_epochs: int,
                 log_interval: int = 200
                 ):
        self.name = name
        self.model = model
        self.train_dl = train_dl
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.n_epochs = n_epochs

    def run(self):
        for epoch in range(self.n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            # TODO: abstract the train epoch method
            for i, dataloader_output in enumerate(self.train_dl):

                # zero the parameter gradients
                self.optimizer.zero_grad()
                loss = self.loss_function(dataloader_output)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

                if i % self.log_interval == self.log_interval - 1:    # print at the log interval
                    print('Epoch: %d / %d, Iteration: %5d, loss: %.3f' %
                          (epoch + 1, self.n_epochs, i + 1, running_loss / self.log_interval))
                    running_loss = 0.0

            # TODO: switch to os.join for safer path creation
            checkpoint_dir = '~/data/' + self.name + '/checkpoint/'
            checkpoint_dir_expanded = os.path.expanduser(checkpoint_dir)
            if not os.path.exists(checkpoint_dir_expanded):
                os.makedirs(checkpoint_dir_expanded)
            checkpoint_file = checkpoint_dir_expanded + str(epoch) + '.pth.tar'
            torch.save(self.model.get_state_dict_all(), checkpoint_file)

            # TODO: generate then run eval code
            # TODO: push results to tensorboardX

        print('Finished Training')
