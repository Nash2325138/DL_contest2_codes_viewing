#!/usr/bin/env python3


import pickle as pkl

from util.data_processor import *


def preprocess(output_filename):
    with open('./data/train_data.pkl', 'rb') as f:
        df = pkl.load(f)

    newdf = preprocess_train_data(df)
    newdf.to_pickle(output_filename)


if __name__ == '__main__':
    output_filename = './data/train_data_one.pkl'
    preprocess(output_filename)
