#!/usr/bin/env python3


import tensorflow as tf

from train.dataset import *
from train.params import *
from train.training import *


def tfsession(growth=None, fraction=None):
    if growth is None and fraction is None:
        return tf.Session()
    elif growth is None and fraction is not None:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        return tf.Session(config=config)
    elif growth is not None and fraction is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = growth
        return tf.Session(config=config)
    else:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        config.gpu_options.allow_growth = growth
        return tf.Session(config=config)


def conv2d(name, input_layer, kernel_size, filters, padding='SAME', relu=True):
    if relu:
        output = tf.layers.conv2d(
                inputs=input_layer,
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                activation=tf.nn.relu,
                name=name)
    else:
        output = tf.layers.conv2d(
                inputs=input_layer,
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                name=name)
    return output


def max_pool(name, input_layer, window):
    return tf.layers.max_pooling2d(
            inputs=input_layer,
            pool_size=[window, window],
            strides=window)


def norm(name, input_layer):
    return tf.layers.batch_normalization(input_layer)


def train_model(sess, model, training_init_op, validation_init_op, next_element, df_train, df_valid, epoch=5):
    for e in range(epoch):
        sess.run(training_init_op)
        losses = []
        while True:
            try:
                x_img, x_img_names = sess.run(next_element)
                x_indx = [
                    df_train.index[df_train['image_name'] == name.decode("utf-8")]
                        .tolist()[0] for name in x_img_names
                ]

                y_gt = get_ground_truth(x_indx, df_train)
                feed_dict = {
                    model.input_layer: x_img,
                    model.gt_bbox_targets: y_gt,
                    model.lr: 0.0001,
                    model.istrain: True
                }

                _, loss, step = sess.run(
                    [model.train_op, model.loss, model.global_step],
                    feed_dict=feed_dict)
                losses.append(loss)

            except tf.errors.OutOfRangeError:
                print('%d epoch with training loss %f' % (e, np.mean(losses)))
                break

        sess.run(validation_init_op)
        losses_v = []
        while True:
            try:
                x_img, x_img_names = sess.run(next_element)
                x_indx = [
                    df_valid.index[df_valid['image_name'] == name.decode("utf-8")]
                    .tolist()[0] for name in x_img_names
                ]
                y_gt = get_ground_truth(x_indx, df_valid)

                feed_dict = {
                    model.input_layer: x_img,
                    model.gt_bbox_targets: y_gt,
                    model.istrain: False
                }

                loss = sess.run([model.loss], feed_dict=feed_dict)

                losses_v.append(loss)
            except tf.errors.OutOfRangeError:
                print('%d epoch with validation loss %f\n' % (e, np.mean(losses_v)))
                break
        if (e + 1) % 5 == 0:
            model.save_model(sess, step, './mycnncheck-{}'.format(e + 1))
    return step


class BasicModel:
    def get_train_op(self):
        pass

    def get_loss_op(self):
        pass

    def train_mode(self):
        pass

    def test_mode(self):
        pass

    def save_model(self, sess, global_step, path):
        pass

    def load_model(self, sess, path):
        pass
