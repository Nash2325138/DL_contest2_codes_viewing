#!/usr/bin/env python3


import numpy as np
import pickle as pkl
import tensorflow as tf

from PIL import Image

from train.params import *
from util.bbox_transform import *


def preprocess_train_data(df):
    width = par_img_width()
    height = par_img_height()
    boxes_resize = df['boxes'].copy()
    for img in range(len(boxes_resize)):
        image = Image.open('/home/Public/JPEGImages/' + df['image_name'][img])
        w = image.size[0]
        h = image.size[1]
        boxes = boxes_resize[img]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (width / w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (height / h)
        boxes_resize[img] = np.array(
                [df['gt_classes'][img][0]] + bbox_transform(
                    np.array([0, 0, width - 1, height - 1]),
                    boxes[0]).tolist())
    new_df = df.copy()
    new_df['one_gt'] = boxes_resize
    return new_df
