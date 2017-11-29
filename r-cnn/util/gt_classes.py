#!/usr/bin/env python3


def get_gt_classes(idx=None):
    classes = [
        '__background__',  # always index 0
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]
    if idx is None:
        return classes
    else:
        return classes[idx]
