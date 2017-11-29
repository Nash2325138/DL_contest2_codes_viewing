#!/usr/bin/env python3


import numpy as np
import pandas as pd
import pickle as pkl
import random

from PIL import Image
from train.params import *
from util.bbox_transform import *


def contains_gt(box, gt):
    bx, by, ba, bb = box
    gx, gy, ga, gb = gt
    return bx <= gx and by <= gy and ga <= ba and gb <= bb


def been_contained(rois, target):
    for i in range(len(rois)):
        if contains_gt(rois[i], target):
            return True
    return False


def try_replace(rois, target):
    for i in range(len(rois)):
        x, y, u, v = rois[i]
        a, b, c, d = target
        if a <= x and b <= y and u <= c and v <= d:
            rois[i] = target
            return True
    return False


def generate_r_cnn_testing(filename, output):
    # target output columns:
    # <image_name>     <roi>
    # image filename   reigon
    df = pd.read_pickle(filename)
    rois = df['rois'].copy()
    total_images = df.shape[0]
    result = []
    for im in range(total_images):
        rois_list = rois[im]
        reduced = []
        random.shuffle(rois_list)
        reduced.append(rois_list[0])
        for i in range(len(rois_list)):
            if been_contained(reduced, rois_list[i]):
                continue
            if try_replace(reduced, rois_list[i]):
                continue
            reduced.append(rois_list[i])
        for i in range(len(reduced)):
            result.append([df['image_name'][im], reduced[i]])
    ndf = pd.DataFrame(result)
    ndf.columns = ['image_name', 'roi']
    ndf.to_pickle(output)


if __name__ == '__main__':
    generate_r_cnn_testing('./local/r-cnn-test-rois-processed.pkl', './local/r-cnn-testing.pkl')
