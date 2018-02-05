import os
import pickle
import inspect
from multiprocessing.pool import Pool

import tensorflow as tf
import numpy as np

from libs.tiny_imagenet_utils import tfrecord_parser
from libs.image_utils import (get_random_patch_list,
                              random_hide,
                              find_location_by_cam)
from libs.various_utils import generate_id_with_date, get_date_time_prefix


def preprocess(image, location, label_one_hot):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [224, 224],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.cast(image, tf.float32)
    image = tf.multiply(image, 1/255.)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image, location, label_one_hot


def preprocess_has(image, location, label_one_hot):
    patch_list = get_random_patch_list(
       image.get_shape().as_list()[0], 32)

    image = tf.py_func(lambda X:  random_hide(X, patch_list),
                       [image],
                       tf.float32)
    return image, location, label_one_hot


def make_dataset_from_tfrecord(tfrecord_path_list,
                               shuffle_buffer_size=5000,
                               batch_size=64,
                               flag_shuffle=False):
    dataset = tf.data.TFRecordDataset(tfrecord_path_list)
    dataset = dataset.map(tfrecord_parser)
    dataset = dataset.map(preprocess)
    if flag_shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset


def make_dataset_from_tfrecord_has(tfrecord_path_list,
                                   shuffle_buffer_size=5000,
                                   batch_size=64,
                                   flag_shuffle=False):
    dataset = tf.data.TFRecordDataset(tfrecord_path_list)
    dataset = dataset.map(tfrecord_parser)
    dataset = dataset.map(preprocess)
    dataset = dataset.map(preprocess_has)
    if flag_shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset


def prepare_data_from_tfrecord(tfrecord_train_dir,
                               tfrecord_valid_dir,
                               tfrecord_test_dir,
                               batch_size=64,
                               shuffle_buffer_size=5000):
    tfrecord_train_path_list = [
        os.path.join(tfrecord_train_dir, tfrecord_name)
        for tfrecord_name in os.listdir(tfrecord_train_dir)]

    tfrecord_valid_path_list = [
        os.path.join(tfrecord_valid_dir, tfrecord_name)
        for tfrecord_name in os.listdir(tfrecord_valid_dir)]

    tfrecord_test_path_list = [
        os.path.join(tfrecord_test_dir, tfrecord_name)
        for tfrecord_name in os.listdir(tfrecord_test_dir)]

    dataset_train = make_dataset_from_tfrecord(
        tfrecord_train_path_list,
        shuffle_buffer_size=shuffle_buffer_size,
        batch_size=batch_size,
        flag_shuffle=True)
    dataset_train_has = make_dataset_from_tfrecord_has(
        tfrecord_train_path_list,
        shuffle_buffer_size=shuffle_buffer_size,
        batch_size=batch_size,
        flag_shuffle=True)
    dataset_valid = make_dataset_from_tfrecord(
        tfrecord_valid_path_list,
        shuffle_buffer_size=shuffle_buffer_size,
        batch_size=batch_size,
        flag_shuffle=False)
    dataset_test = make_dataset_from_tfrecord(
        tfrecord_test_path_list,
        shuffle_buffer_size=shuffle_buffer_size,
        batch_size=batch_size,
        flag_shuffle=False)

    iterator = tf.data.Iterator.from_structure(
        dataset_train.output_types, dataset_train.output_shapes)

    init_dataset_train = iterator.make_initializer(dataset_train)
    init_dataset_train_has = iterator.make_initializer(dataset_train_has)
    init_dataset_valid = iterator.make_initializer(dataset_valid)
    init_dataset_test = iterator.make_initializer(dataset_test)

    X, P, Y = iterator.get_next()

    data_dict = {
        'X': X,
        'P': P,
        'Y': Y,
        'iterator': iterator,
        'init_dataset_train': init_dataset_train,
        'init_dataset_train_has': init_dataset_train_has,
        'init_dataset_valid': init_dataset_valid,
        'init_dataset_test': init_dataset_test,
    }
    return data_dict 

