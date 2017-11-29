#!/usr/bin/env python3


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


def ratio_intersection_over_union(rect1, rect2, verbose=0):
    rect1 = np.array(rect1, dtype=np.float32)
    rect2 = np.array(rect2, dtype=np.float32)
    X_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    Y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))

    def area(rect):
        return (rect[2] - rect[0]) * (rect[3] - rect[1])

    a_inter = X_overlap * Y_overlap
    a_union = area(rect1) + area(rect2) - a_inter
    if verbose == 1:
        print(area(rect1), area(rect2))
        print(a_inter)
        print(a_union)
    return a_inter / a_union


def generate_r_cnn_training(filename, output):
    # target output columns:
    # <image_name>     <gt_class>   <gt_box>     <roi>      <gt_box_norm>
    # image filename   class        ground       reigon     ground
    #                               truth box    of         truth box
    #                               (w/o norm)   interest   (shifted)
    #                                            (w/o norm)
    df = pd.read_pickle(filename)
    rois = df['rois'].copy()
    gt_boxes = df['boxes'].copy()
    total_images = df.shape[0]
    result = []

    truth_threshold = 2
    bg_threshold = 1
    failed = 0

    for im in range(total_images):
        if im % 2000 == 0:
            print('Processing... {}/{}'.format(im, total_images))
        rois_list = rois[im]
        random.shuffle(rois_list)
        for i in range(len(gt_boxes[im])):
            truth_found = 0
            bg_found = 1
            for j in range(len(rois_list)):
                if ratio_intersection_over_union(rois_list[j], gt_boxes[im][i]) >= 0.5:
                    if truth_found < truth_threshold:
                        truth_found += 1
                        gt_one = [df['gt_classes'][im][i]]
                        gt_one.extend(calculate_gt_one(rois_list[j], gt_boxes[im][i].tolist()))
                        result.append([
                            df['image_name'][im],
                            df['gt_classes'][im][i],
                            gt_boxes[im][i].tolist(),
                            rois_list[j],
                            gt_one
                        ])
                elif ratio_intersection_over_union(rois_list[j], gt_boxes[im][i]) > 0.15:
                    if bg_found < bg_threshold:
                        bg_found += 1
                        gt_one = [0]
                        gt_one.extend(calculate_gt_one(rois_list[j], rois_list[j]))
                        result.append([
                            df['image_name'][im],
                            0,
                            rois_list[j],
                            rois_list[j],
                            gt_one
                        ])
                if truth_found >= truth_threshold and bg_found >= truth_threshold:
                    break
            if truth_found < truth_threshold or bg_found < bg_threshold:
                failed += 1

    ndf = pd.DataFrame(result)
    ndf.columns = ['image_name', 'gt_class', 'gt_box', 'roi', 'gt_one']
    ndf.to_pickle(output)
    print('\n Summary: failed: {}'.format(failed))


if __name__ == '__main__':
    generate_r_cnn_training('./local/r-cnn-rois-processed.pkl', './local/r-cnn-training.pkl')
