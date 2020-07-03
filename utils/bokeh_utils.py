from collections import OrderedDict
from typing import List

import numpy as np
import cv2 as cv

from PIL import Image

from bokeh.plotting import output_file, show, figure


# TODO: add typing and include torch plotting capability as well as asserts for security. Also check on the margin issue.

def image_report(imgs: OrderedDict, link_plots: bool = False):
    '''
    Produces a list of bokeh figures containing visualized images
    :param imgs: Ordered Dict with keys containing image titles and values containing numpy arrays
        containing the images. Both greyscale and rgb images are accepted.
    :param
    :return: a list of bokeh figures
    '''

    figs = []

    for title, image in imgs.items():


        ydim = image.shape[0]
        xdim = image.shape[1]
        dim = max(xdim, ydim)

        if link_plots and len(figs) > 0:
            fig = figure(title=title, x_range=figs[0].x_range, y_range=figs[0].y_range, active_scroll="wheel_zoom")
        else:
            fig = figure(title=title, x_range=(0, dim), y_range=(0, dim), active_scroll="wheel_zoom")

        if len(image.shape) > 2:
            # color images
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = Image.fromarray(image).convert('RGBA')
            xdim, ydim = image.size
            # Create an array representation for the image `img`, and an 8-bit "4
            # layer/RGBA" version of it `view`.
            img = np.empty((ydim, xdim), dtype=np.uint32)
            view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
            view[:, :, :] = np.flipud(np.asarray(image))

            # Display the 32-bit RGBA image
            # TODO: readdress the sizing issue by placing a plot_width arg
            fig.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)

        else:
            # greyscale images
            fig.image(image=[np.flipud(image)],
                      x=[0], y=[0],
                      dw=[xdim], dh=[ydim])

        figs.append(fig)

    return figs
