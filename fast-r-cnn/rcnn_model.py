#!/usr/bin/env python3


import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf

from train.rcnn import *
from train.dataset import *
from train.params import *
from train.training import *


def main():
    batch_size = par_batch_size()
    img_width = par_img_width()
    img_height = par_img_height()
    num_classes = par_num_classes()

    df = pd.read_pickle('./local/r-cnn-training.pkl')
    df_train, df_valid = train_valid_split(df, 0.1)

    tf.reset_default_graph()

    X_train_images = tf.constant(df_train['image_name'].as_matrix())
    X_valid_images = tf.constant(df_valid['image_name'].as_matrix())

    X_train_rois = tf.constant(list(df_train['roi'].as_matrix()))
    X_valid_rois = tf.constant(list(df_valid['roi'].as_matrix()))

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_images, X_train_rois))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid_images, X_valid_rois))

    train_dataset = train_dataset.map(data_generator, num_parallel_calls=16)
    valid_dataset = valid_dataset.map(data_generator, num_parallel_calls=16)

    train_dataset = train_dataset.prefetch(8 * batch_size)
    valid_dataset = valid_dataset.prefetch(8 * batch_size)

    train_dataset = train_dataset.shuffle(8 * batch_size)
    valid_dataset = valid_dataset.shuffle(8 * batch_size)

    train_dataset = train_dataset.batch(batch_size)
    valid_dataset = valid_dataset.batch(batch_size)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init = iterator.make_initializer(train_dataset)
    valid_init = iterator.make_initializer(valid_dataset)

    model = RCNNModel()

    with tfsession() as sess:
        sess.run(tf.global_variables_initializer())
        step = train_model(sess, model, training_init, valid_init, next_element, df_train, df_valid, epoch=10)
        model.save_model(sess, step, './myrcnncheck')


if __name__ == '__main__':
    main()
