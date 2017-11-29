#!/usr/bin/env python3


import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf

from util.data_processor import *
from train.cnn import *
from train.dataset import *
from train.params import *
from train.training import *


def main():
    batch_size = par_batch_size()
    img_width = par_img_width()
    img_height = par_img_height()
    num_classes = par_num_classes()

    with open('./data/train_data_one.pkl', 'rb') as f:
        df = pkl.load(f)
    df_train, df_valid = train_valid_split(df, 0.1)
    df_train = df

    tf.reset_default_graph()

    X_train_filename = tf.constant(df_train['image_name'].as_matrix())
    X_valid_filename = tf.constant(df_valid['image_name'].as_matrix())

    train_dataset = tf.data.Dataset.from_tensor_slices(X_train_filename)
    valid_dataset = tf.data.Dataset.from_tensor_slices(X_valid_filename)

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

    model = CNNModel()

    with tfsession() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists('./mycnncheck'):
            model.load_model(sess, './mycnncheck')
        step = train_model(sess, model, training_init, valid_init, next_element, df_train, df_valid, epoch=25)
        model.save_model(sess, step, './mycnncheck')


if __name__ == '__main__':
    main()
