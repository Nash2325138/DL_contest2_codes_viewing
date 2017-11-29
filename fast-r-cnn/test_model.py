#!/usr/bin/env python3


import os
import pickle as pkl
import sys
import tensorflow as tf

from util.bbox_transform import *
from train.dataset import *
from train.rcnn import *
from train.evaluation import *
from train.params import *
from train.training import *


def test_model():
    df_test = pd.read_pickle('./local/r-cnn-testing.pkl')

    tf.reset_default_graph()

    X_test_image = tf.constant(df_test['image_name'].as_matrix())
    X_test_roi = tf.constant(list(df_test['roi'].as_matrix()))

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_image, X_test_roi))

    test_dataset = test_dataset.map(data_generator, num_parallel_calls=16)
    test_dataset.prefetch(20)
    test_dataset = test_dataset.batch(1)

    iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    next_element = iterator.get_next()

    testing_init = iterator.make_initializer(test_dataset)

    model = RCNNModel()

    result_cls, result_reg, result_img_name = [], [], []
    test_cls_dict = dict()
    # key: image_id, value: a set with cls excluding zero (background)

    img_id = 0
    img_id_dict = dict()

    total_img = df_test.shape[0]
    running_idx = 0

    with tfsession() as sess:
        model.load_model(sess, './myrcnncheck-2')
        sess.run(testing_init)
        while True:
            try:
                x_img, x_img_name = sess.run(next_element)

                feed_dict = {
                    model.input_layer: x_img,
                    model.istrain: False
                }

                logits_cls, logits_reg = sess.run(
                    [model.out_cls, model.logits_reg], feed_dict=feed_dict)
                result_cls.append(logits_cls)
                result_reg.append(logits_reg)
                print(x_img_name, logits_cls, logits_reg)
                sys.exit(1)
                img_name = x_img_name[0].decode('utf-8')
                result_img_name.append(img_name)
                if img_name not in img_id_dict:
                    img_id_dict[img_name] = img_id
                    img_id += 1
                    test_cls_dict[img_name] = set()
                running_idx += 1
                if running_idx % 2000 == 0:
                    print('Running {}/{}'.format(running_idx, total_img))
            except tf.errors.OutOfRangeError:
                break

    num_test_img = df_test.shape[0]
    bbox_preds = [[] for i in range(num_test_img)]
    bbox_cls = [[] for i in range(num_test_img)]
    img_width = par_img_width()
    img_height = par_img_height()
    for idx in range(len(result_cls)):
        img_cls = np.argmax(result_cls[idx])
        if img_cls == 0:
            continue
        if img_cls in test_cls_dict[result_img_name[idx]]:
            continue
        test_cls_dict[result_img_name[idx]].add(img_cls)
        img_id = img_id_dict[result_img_name[idx]]
        bbox_pred = calculate_reg_box(df_test['roi'][idx], result_reg[idx][0])
        bbox_c = np.argmax(result_cls[idx])

        bbox_cls[img_id].append(bbox_c)
        bbox_preds[img_id].append(np.array(bbox_pred))

    fbbox_cls = []
    fbbox_preds = []
    for element in bbox_cls:
        fbbox_cls.append(np.array(element))
    for element in bbox_preds:
        fbbox_preds.append(np.array(element))

    evaluate(fbbox_preds, fbbox_cls)


if __name__ == '__main__':
    test_model()
