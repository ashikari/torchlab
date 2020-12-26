import os

import numpy as np
import torch
import torch.nn.functional as F

from utils.logger import logger

from projects.cifar_ex.runtime import CifarLab
from projects.cifar_ex.data.dataloaders import get_loader


# setup dataloader
dataloader = get_loader(batch_size=1, num_workers=1, split='test')
dataiter = iter(dataloader)

# instantiate model
model = CifarLab.get_model()

if torch.cuda.is_available():
    model.cuda()

# load weights into model

# TODO: make the number of epochs dependent on the available checkpoints (use GLOB)
for epoch in range(5):
    checkpoint_dir = '~/data/trial_run/checkpoint/'
    checkpoint_dir_expanded = os.path.expanduser(checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir_expanded + str(epoch) + '.pth.tar')
    model.load_state_dict(checkpoint)

    num_imgs = 1000

    with torch.no_grad():
        score = 0
        for i in range(num_imgs):

            image, label = dataiter.next()
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            output = F.softmax(model(image))
            label_est = np.argmax(output.cpu().numpy())

            if label.item() == label_est:
                score += 1

        print("Accuarcy: {:.2}% @ Epoch: {}".format(
            score / num_imgs, epoch))
