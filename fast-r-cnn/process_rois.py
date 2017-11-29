#!/usr/bin/env python3


import numpy as np
import pandas as pd
import pickle as pkl
import random


def decrease_rois(inputfile, outputfile):
    df = pd.read_pickle(inputfile)
    rois = df['selective_search_boxes'].copy()
    for i in range(len(rois)):
        roislist = np.array(rois[i]).tolist()
        random.shuffle(roislist)
        rois[i] = roislist
        #  roislist = sorted(roislist)
        #  my_rois = {}
        #  for j in range(len(roislist)):
        #      x1, y1, x2, y2 = roislist[j]
        #      if (x1, y1) in my_rois:
        #          cur_x2, cur_y2 = my_rois[(x1, y1)]
        #          cur_x2 = max(cur_x2, x2)
        #          cur_y2 = max(cur_y2, y2)
        #          my_rois[(x1, y1)] = (cur_x2, cur_y2)
        #      else:
        #          my_rois[(x1, y1)] = (x2, y2)
        #  xys = []
        #  for x1, y1 in my_rois:
        #      x2, y2 = my_rois[(x1, y1)]
        #      xys.append([x1, y1, x2, y2])
        #  rois[i] = xys
    df['rois'] = rois
    df.to_pickle(outputfile)


if __name__ == '__main__':
    decrease_rois('./data/train_data.pkl', './local/r-cnn-rois-processed.pkl')
    decrease_rois('./data/test_data.pkl', './local/r-cnn-test-rois-processed.pkl')
