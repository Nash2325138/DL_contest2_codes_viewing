#!/usr/bin/env python3


import numpy as np
import os
import sys
import tensorflow as tf

from train.dataset import *
from train.params import *
from train.training import *


class CNNModel(BasicModel):
    def __init__(self, name='CNNModel'):
        self.name = name
        self.istrain = tf.placeholder(tf.bool, shape=[])
        with tf.variable_scope(self.name):
            self.build()

    def build(self):
        # constants
        batch_size = par_batch_size()
        img_width = par_img_width()
        img_height = par_img_height()
        num_classes = par_num_classes()

        # input image and roiboxes
        self.input_layer = tf.placeholder(
            dtype=tf.float32, shape=[None, img_width, img_height, 3])
        # input training ground truth [batch_number, [label, 4]]
        self.gt_bbox_targets = tf.placeholder(dtype=tf.float32, shape=[None, 5])

        # conv 1_1
        conv1_1 = conv2d('conv1_1', self.input_layer, [3, 3], 64)
        # conv 1_2
        conv1_2 = conv2d('conv1_2', conv1_1, [3, 3], 64)
        # pool 1
        pool1 = max_pool('pool1', conv1_2, 2)
        # norm 1
        norm1 = norm('norm1', pool1)

        # conv 2_1
        conv2_1 = conv2d('conv2_1', norm1, [3, 3], 64)
        # conv 2_2
        conv2_2 = conv2d('conv2_2', conv2_1, [3, 3], 64)
        # pool 2
        pool2 = max_pool('pool2', conv2_2, 2)
        # norm 2
        norm2 = norm('norm2', pool2)

        # conv 3_1
        conv3_1 = conv2d('conv3_1', norm2, [3, 3], 64)
        # pool 3
        pool3 = max_pool('pool3', conv3_1, 2)
        # norm 3
        norm3 = norm('norm3', pool3)

        # conv 4_1
        conv4_1 = conv2d('conv4_1', norm3, [3, 3], 64)
        # pool 4
        pool4 = max_pool('pool4', conv4_1, 2)
        # norm 4
        norm4 = norm('norm4', pool4)
        # conv 4_2
        conv4_2 = conv2d('conv4_2', norm4, [3, 3], 64)
        # pool 5
        pool5 = max_pool('pool5', conv4_2, 4)
        # norm 5
        norm5 = norm('norm5', pool5)

        print(norm5)
        flatten = tf.reshape(norm5, [-1, 1792])
        print(flatten)

        # dense layers
        dense1 = tf.layers.dense(flatten, 256, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(dense1, rate=0.5, training=self.istrain)

        dense2 = tf.layers.dense(dropout1, 256, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(dense2, rate=0.5, training=self.istrain)

        # box and class predication
        # for object classification
        self.logits_cls = tf.layers.dense(dropout2, num_classes)
        self.out_cls = tf.nn.softmax(self.logits_cls)

        # for bounding box prediction
        self.logits_reg = tf.layers.dense(dropout2, 4)

        # calculate loss
        gt_cls, gt_reg = tf.split(self.gt_bbox_targets, [1, 4], 1)

        gt_cls_raw = tf.cast(gt_cls, tf.int64)
        gt_cls_raw_float = tf.where(gt_cls_raw == 0, 0.0, 1.0)
        gt_cls = tf.reshape(tf.one_hot(gt_cls_raw, num_classes), [-1, num_classes])

        self.loss_cls = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=gt_cls, logits=self.logits_cls))

        self.reg_delta = self.logits_reg - gt_reg
        self.reg_delta_square = tf.multiply(self.reg_delta, self.reg_delta)
        self.reg_delta_square = tf.multiply(self.reg_delta_square, 0.5)
        self.reg_delta_abs = tf.abs(self.reg_delta)
        self.loss_reg_sm = tf.where(
                self.reg_delta_abs < 1,
                self.reg_delta_square,
                tf.subtract(self.reg_delta_abs, 0.5))
        self.loss_reg = tf.reduce_mean(tf.reduce_sum(self.loss_reg_sm))
        #  self.loss_reg = tf.losses.mean_squared_error(gt_reg, self.logits_reg)

        self.loss = self.loss_cls + gt_cls_raw_float * self.loss_reg

        self.lr = tf.placeholder(tf.float32, [])
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def load_model(self, sess, path):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        print('Loading model {}.'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

    def save_model(self, sess, global_step, path):
        saver = tf.train.Saver()
        fullname = os.path.join(path, 'cnn')
        saver.save(sess, fullname, global_step)
