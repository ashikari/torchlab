import os

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from bokeh.plotting import output_file, show
from bokeh.layouts import layout

from projects.cifar_ex.data.dataloaders import get_train_loader
from utils.bokeh_utils import image_report


# TODO: Create CLI for this file -- batch size only
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device: {}".format(device))

train_dl = get_train_loader(batch_size=4)

dataiter = iter(train_dl)

batch_inputs, batch_labels = dataiter.next()

# untransform the image and plot
# TODO: place this code into the image report function
# TODO: define untransfom in the dataloader
images = [255 * (img.numpy() / 2 + 0.5) for img in batch_inputs]

# change encoding for visualization
# TODO: place this code into the image report function
images = {str(i): np.transpose(npimg, (1, 2, 0)).astype("uint8")
          for i, npimg in enumerate(images)}
fig = image_report(images)

dataloader_viz_report = layout(fig)
data_dir = '~/data/'
data_dir_expanded = os.path.expanduser(data_dir)
output_file(data_dir_expanded + "reports/cifar_data_viz.html")
show(dataloader_viz_report)
