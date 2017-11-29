#!/usr/bin/env python3


def get_hparams():
    params = {
            img_width: par_img_width(),
            img_height: par_img_height(),
            batch_size: par_batch_size(),
            num_classes: par_num_classes()
    }
    return params


def par_img_width():
    return 500


def par_img_height():
    return 300


def par_batch_size():
    return 16


def par_num_classes():
    return 21
