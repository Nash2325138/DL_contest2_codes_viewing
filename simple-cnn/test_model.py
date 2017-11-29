#!/usr/bin/env python3


import os
import pickle as pkl
import tensorflow as tf

from util.data_processor import *
from train.dataset import *
from train.cnn import *
from train.evaluation import *
from train.params import *
from train.training import *


def test_model():
    df_test = pd.read_pickle('./data/test_data.pkl')

    tf.reset_default_graph()

    X_test_image = tf.constant(df_test['image_name'].as_matrix())
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_image))

    test_dataset = test_dataset.map(
            data_generator)
    test_dataset = test_dataset.batch(1)

    iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    next_element = iterator.get_next()

    testing_init = iterator.make_initializer(test_dataset)

    model = CNNModel()

    results_dict = dict()
    result_cls, result_reg = [], []

    with tfsession() as sess:
        model.load_model(sess, './mycnncheck-5')
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
                results_dict[x_img_name[0].decode('utf-8')] = (logits_cls, logits_reg)
            except tf.errors.OutOfRangeError:
                break

    for k in df_test['image_name']:
        v = results_dict[k]
        result_cls.append(v[0])
        result_reg.append(v[1])

    with open('result_cls.pkl', 'wb') as f:
        pkl.dump(result_cls, f)

    num_test_img = df_test.shape[0]
    bbox_preds = []
    bbox_cls = []
    img_width = par_img_width()
    img_height = par_img_height()
    for img in range(num_test_img):
        bbox_pred = []
        bbox_c = []
        bbox_pred.append(
            reg_to_bbox(result_reg[img][0], np.array([0, 0, img_width, img_height])))
        bbox_c.append(np.argmax(result_cls[img]))

        bbox_cls.append(np.array(bbox_c))
        bbox_preds.append(np.array(bbox_pred))

    for img in range(num_test_img):
        imgage = Image.open("./data/JPEGImages/" + df_test['image_name'][img])
        w = imgage.size[0]
        h = imgage.size[1]
        boxes = bbox_preds[img]

        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (w / img_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (h / img_height)
        bbox_preds[img] = boxes

    evaluate(bbox_preds, bbox_cls)


if __name__ == '__main__':
    test_model()
