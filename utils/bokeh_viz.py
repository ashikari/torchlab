from collections import OrderedDict

import numpy as np
import cv2 as cv

from PIL import Image

from bokeh.plotting import output_file, show, figure
from bokeh.layouts import layout
from bokeh.models.widgets import Panel, Tabs

from utils.bokeh_utils import image_report
from utils.logger import logger

# TODO: input paths from CLI or auxiliary config file or something
data_dir = '~/data/'
data_dir_expanded = os.path.expanduser(data_dir)
img_paths = [data_dir_expanded + '/datasets/test_imgs/prettywater.jpg']
tabs = []

log = logger()

for i, img_path in enumerate(img_paths):
    with log.log_runtime("read"):
        img = cv.imread(img_path)

    bin_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    images = OrderedDict()

    images["original"] = img
    images["binary"] = bin_img

    with log.log_runtime("report"):
        figs = image_report(images)
    tabs.append(Panel(child=layout([figs]), title=str(i)))

output_file(data_dir_expanded + "/reports/bokeh_test.html")
show(Tabs(tabs=tabs))

log.export(data_dir_expanded + '/log_test.txt')
