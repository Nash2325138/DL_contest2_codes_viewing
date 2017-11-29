#!/usr/bin/env python3


import pandas as pd
import pickle as pkl
import sys

from PIL import Image


def crop_rois(df, classes_filename):
    f = open(classes_filename, 'w')
    f.write('filename,class\n')
    for img in range(len(df)):
        filename = df['image_name'][img]
        image = Image.open('/home/Public/JPEGImages/' + df['image_name'][img])
        classes = df['gt_classes'][img]
        boxes = df['boxes'][img]
        for idx, box in enumerate(boxes):
            newimg = image.crop(box)
            newclass = classes[idx]
            nfile = filename.replace('.jpg', '_{}.jpg'.format(idx))
            f.write('{},{}\n'.format(nfile, newclass))
            newimg.save('./croppedImages/' + nfile)
    f.close()
