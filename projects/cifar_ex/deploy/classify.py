import os

from glob import glob
import numpy as np
import torch
import torch.nn.functional as F

from utils.logger import logger

from projects.cifar_ex.runtime import CifarLab
from projects.cifar_ex.data.dataloaders import get_loader


# setup dataloader
batch_size = 10
dataloader = get_loader(batch_size=batch_size, num_workers=1, split='test')
dataiter = iter(dataloader)

# instantiate model
model = CifarLab.get_model()

if torch.cuda.is_available():
    model.cuda()

# grab checkpoint paths
checkpoint_dir = '~/data/trial_run/checkpoint/'
checkpoint_dir_expanded = os.path.expanduser(checkpoint_dir)
checkpoint_paths = glob(checkpoint_dir_expanded + "*")

for epoch, checkpoint_path in enumerate(checkpoint_paths):
    # load weights into model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    num_batches = 100

    with torch.no_grad():
        score = 0
        num_images = 0
        for i in range(num_batches):

            image, label = dataiter.next()
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            output = model.predict(image)
            label_est = torch.argmax(output, axis=-1)

            score += torch.sum(label_est==label)
            num_images += label.shape[0]

        score = score.cpu().item()
        print("Accuarcy: {:.2}% @ Epoch: {}, num_images: {}".format(
            score / num_batches * batch_size, epoch, num_images))
