#!/usr/bin/env python3


import numpy as np
import random
import tensorflow as tf

from train.params import *


def data_generator(image_filename):
    file_path = '/home/Public/JPEGImages/'
    image_width = par_img_width()
    image_height = par_img_height()
    img_path = file_path + image_filename
    img_file = tf.read_file(img_path)
    img = tf.image.decode_image(img_file, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])
    img = tf.image.resize_images(img, size=[image_width, image_height])
    return img, image_filename


def get_ground_truth(x_indices, dataframe):
    target_batch = []
    for index in x_indices:
        target_batch.append(dataframe['one_gt'][index])
    return np.array(target_batch)


def train_valid_split(df, valid_size):
    valid_random = np.random.rand(len(df)) < valid_size
    return df[~valid_random].reset_index(drop=True), df[valid_random].reset_index(drop=True)
