#!/usr/bin/env python3


import numpy as np
import random
import tensorflow as tf

from train.params import *


def data_generator(image_filename, image_roi):
    img = image_generator(image_filename)
    img = crop_resize_image(img, image_roi)
    return img, image_filename


def image_generator(image_filename):
    file_path = '/home/Public/JPEGImages/'
    img_path = file_path + image_filename
    img_file = tf.read_file(img_path)
    img = tf.image.decode_image(img_file, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])
    return img


def crop_resize_image(image, roi):
    width = par_img_width()
    height = par_img_height()
    offset_width = roi[0]
    offset_height = roi[1]
    target_width = roi[2] - roi[0]
    target_height = roi[3] - roi[1]
    img = tf.image.crop_to_bounding_box(
            image,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=target_height,
            target_width=target_width)
    img = tf.image.resize_images(img, [height, width])
    return img


def get_ground_truth(x_indices, dataframe):
    target_batch = []
    for index in x_indices:
        target_batch.append(dataframe['gt_one'][index])
    return np.array(target_batch)


def train_valid_split(df, valid_size):
    valid_random = np.random.rand(len(df)) < valid_size
    return df[~valid_random].reset_index(drop=True), df[valid_random].reset_index(drop=True)
