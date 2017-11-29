#!/usr/bin/env python3


import numpy as np

from train.params import *


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[2] - ex_rois[0] + 1.0
    ex_heights = ex_rois[3] - ex_rois[1] + 1.0
    ex_ctr_x = ex_rois[0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[1] + 0.5 * ex_heights

    gt_widths = gt_rois[2] - gt_rois[0] + 1.0
    gt_heights = gt_rois[3] - gt_rois[1] + 1.0
    gt_ctr_x = gt_rois[0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.array([targets_dx, targets_dy, targets_dw, targets_dh])
    return targets


def reg_to_bbox(reg, box):
    img_width = box[2] - box[0]
    img_height = box[3] - box[1]

    bbox_width = box[2] - box[0] + 1.0
    bbox_height = box[3] - box[1] + 1.0
    bbox_ctr_x = box[0] + 0.5 * bbox_width
    bbox_ctr_y = box[1] + 0.5 * bbox_height

    out_ctr_x = reg[0] * bbox_width + bbox_ctr_x
    out_ctr_y = reg[1] * bbox_height + bbox_ctr_y

    out_width = bbox_width * np.exp(reg[2])
    out_height = bbox_height * np.exp(reg[3])

    return np.array([
        max(0, out_ctr_x - 0.5 * out_width),
        max(0, out_ctr_y - 0.5 * out_height),
        min(img_width, out_ctr_x + 0.5 * out_width),
        min(img_height, out_ctr_y + 0.5 * out_height)
    ])


def transform_reigons(roi, box, target):
    reigon_w = target[2] - target[0]
    reigon_h = target[3] - target[1]
    x, y, u, v = roi
    a, b, c, d = box
    # move roi to (0, 0) first
    # and record delta, so box -= delta
    x_delta = x
    y_delta = y
    x, y = 0, 0
    u -= x_delta
    v -= y_delta
    a -= x_delta
    b -= y_delta
    c -= x_delta
    d -= y_delta
    # resize_coef = (reigon_w / u), (reigon_h / v)
    coef_x = reigon_w / u
    coef_y = reigon_h / v
    u *= coef_x
    v *= coef_y
    a *= coef_x
    b *= coef_y
    c *= coef_x
    d *= coef_y
    # transform to target
    x += target[0]
    y += target[1]
    u += target[0]
    v += target[1]
    a += target[0]
    b += target[1]
    c += target[0]
    d += target[1]
    return [x, y, u, v], [a, b, c, d]


def calculate_gt_one(roi, gt_box):
    nroi, nbox = transform_reigons(roi, gt_box, [0, 0, par_img_width(), par_img_height()])
    return bbox_transform(nroi, nbox)


def calculate_reg_box(ori_roi, reg_box):
    nbox = reg_to_bbox(reg_box, [0, 0, par_img_width(), par_img_height()])
    roi, box = transform_reigons([0, 0, par_img_width(), par_img_height()], nbox, ori_roi)
    return box
