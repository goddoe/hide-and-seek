import os

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.ndimage import imread

from libs.utils import split_train_valid


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def idx_list_to_one_hot_list(idx_list, n_class):
    # Y to one hot
    n_sample = len(idx_list)
    tmp = np.zeros((n_sample, n_class))
    tmp[np.arange(n_sample), idx_list] = 1
    one_hot_list = tmp

    return one_hot_list


def read_tiny_imagenet(path, train_ratio=0.8):
    """
    In this test we use validation set as test set
    """
    # Path
    words_path = "{}/words.txt".format(path)
    train_path = "{}/train".format(path)
    test_path = "{}/val".format(path)

    # Get tiny imagenet dict
    idx_word_dict = {}
    word_idx_dict = {}

    idx_nid_dict = {}
    nid_idx_dict = {}

    nid_used = os.listdir(train_path)
    with open(words_path, "rt") as f:
        line_list = f.readlines()

    idx = 0
    for line in line_list:
        nid, word = line.split('\t')
        word = "{}_{}".format(idx, word.strip())
        nid = nid.strip()
        if nid not in nid_used:
            continue
        idx_word_dict[idx] = word
        word_idx_dict[word] = idx

        idx_nid_dict[idx] = nid
        nid_idx_dict[nid] = idx
        idx += 1

    # Read data
    train_dir_path_list = [os.path.join(train_path, name) for name in os.listdir(train_path)]

    X_train_valid = []
    Y_train_valid = []
    P_train_valid = []

    # Read train, valid
    for train_dir_path in train_dir_path_list:
        loc_path = "{}/{}_boxes.txt".format(train_dir_path, os.path.basename(train_dir_path))
        img_dir_path = "{}/images".format(train_dir_path)

        img_path_list = [os.path.join(img_dir_path, name) for name in os.listdir(img_dir_path)]

        with open(loc_path, "rt") as f:
            loc_line_list = f.readlines()
        file_name_loc_dict = {}
        for loc_line in loc_line_list:
            (
                file_name,
                pos_x,
                pos_y,
                pos_w,
                pos_h
            ) = [
                    val.strip()
                    if i == 0 else int(val.strip())
                    for i, val in enumerate(loc_line.split('\t'))
                ]

            file_name_loc_dict[file_name] = [pos_x, pos_y, pos_w, pos_h]

        label = nid_idx_dict[os.path.basename(train_dir_path)]
        for img_path in img_path_list:
            file_name = os.path.basename(img_path)

            img = imread(img_path, mode='RGB')
            X_train_valid.append(img)
            Y_train_valid.append(label)
            P_train_valid.append(file_name_loc_dict[file_name])

    X_train_valid = np.stack(X_train_valid)
    Y_train_valid = np.stack(Y_train_valid)
    P_train_valid = np.stack(P_train_valid)

    (train_data_list,
     valid_data_list) = split_train_valid(data_list=[X_train_valid,
                                                     Y_train_valid,
                                                     P_train_valid],
                                          train_ratio=0.8,
                                          flag_random=True)

    X_train = train_data_list[0]
    Y_train = train_data_list[1]
    P_train = train_data_list[2]

    X_valid = valid_data_list[0]
    Y_valid = valid_data_list[1]
    P_valid = valid_data_list[2]

    # Read Test
    img_dir_path = "{}/images".format(test_path)
    annotation_path = "{}/val_annotations.txt".format(test_path)

    annotation_pd = pd.read_csv(annotation_path, header=None, sep='\t', index_col=0)
    img_path_list = [os.path.join(img_dir_path, file_name) for file_name in os.listdir(img_dir_path)]

    X_test = []
    Y_test = []
    P_test = []

    for img_path in img_path_list:
        img = imread(img_path, mode='RGB')
        nid = annotation_pd[1][os.path.basename(img_path)]
        label = nid_idx_dict[nid]
        pos_x = annotation_pd[2][os.path.basename(img_path)]
        pos_y = annotation_pd[3][os.path.basename(img_path)]
        pos_w = annotation_pd[4][os.path.basename(img_path)]
        pos_h = annotation_pd[5][os.path.basename(img_path)]

        X_test.append(img)
        Y_test.append(label)
        P_test.append([pos_x, pos_y, pos_w, pos_h])

    X_test = np.stack(X_test)
    Y_test = np.stack(Y_test)
    P_test = np.stack(P_test)

    Y_train_one_hot = idx_list_to_one_hot_list(Y_train, len(idx_nid_dict))
    Y_valid_one_hot = idx_list_to_one_hot_list(Y_valid, len(idx_nid_dict))
    Y_test_one_hot = idx_list_to_one_hot_list(Y_test, len(idx_nid_dict))

    result_dict = {
            'X_train': X_train,
            'Y_train': Y_train,
            'Y_train_one_hot': Y_train_one_hot,
            'P_train': P_train,

            'X_valid': X_valid,
            'Y_valid': Y_valid,
            'Y_valid_one_hot': Y_valid_one_hot,
            'P_valid': P_valid,

            'X_test': X_test,
            'Y_test': Y_test,
            'Y_test_one_hot': Y_test_one_hot,
            'P_test': P_test,

            'idx_word_dict': idx_word_dict,
            'word_idx_dict': word_idx_dict,
            'idx_nid_dict': idx_nid_dict,
            'nid_idx_dict': nid_idx_dict,
        }

    return result_dict


def save_with_tfrecord(tfrecord_dir, X, Y, Y_one_hot, P, label_depth, shard_size=1000, prefix=''):
    """
    Save as Tfrecord
    """
    n_shard = len(X)//shard_size
    n_shard += 0 if len(X) % shard_size == 0 else 1

    tfrecord_path_list = []

    try: os.makedirs(tfrecord_dir)
    except: pass

    for shard_i in range(n_shard):
        shard_X = X[
            shard_i * shard_size: (shard_i + 1) * shard_size]

        shard_P = P[
            shard_i * shard_size: (shard_i + 1) * shard_size]

        shard_Y = Y[
            shard_i * shard_size: (shard_i + 1) * shard_size]

        shard_Y_one_hot = Y_one_hot[
            shard_i * shard_size: (shard_i + 1) * shard_size]

        tfrecord_filename = '{}/{}_{:05.0f}_of_{:05.0f}.tfrecords'.format(
            tfrecord_dir, prefix, shard_i, n_shard)

        with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
            for i in range(len(shard_X)):
                img = shard_X[i]
                location = shard_P[i]
                label = int(shard_Y[i])
                label_one_hot = shard_Y_one_hot[i]

                (height, width, channel) = img.shape

                img_raw = img.tostring()
                location_raw = location.tostring()
                label_one_hot_raw = label_one_hot.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'channel': _int64_feature(channel),
                    'label': _int64_feature(label),
                    'label_depth': _int64_feature(label),

                    'label_one_hot_raw': _bytes_feature(label_one_hot_raw),
                    'image_raw': _bytes_feature(img_raw),
                    'location_raw': _bytes_feature(location_raw)}))
                writer.write(example.SerializeToString())

        tfrecord_path_list.append(tfrecord_filename)
    return tfrecord_path_list


def tfrecord_parser(record):
    parsed = tf.parse_single_example(
        record,
        features={
            'height': tf.FixedLenFeature((), tf.int64),
            'width': tf.FixedLenFeature((), tf.int64),
            'channel': tf.FixedLenFeature((), tf.int64),
            'label': tf.FixedLenFeature((), tf.int64),
            'label_depth': tf.FixedLenFeature((), tf.int64),

            'label_one_hot_raw': tf.FixedLenFeature((), tf.string),
            'image_raw': tf.FixedLenFeature((), tf.string),
            'location_raw': tf.FixedLenFeature((), tf.string)})

    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.reshape(image, [64, 64, 3])
    label_one_hot = tf.decode_raw(parsed['label_one_hot_raw'], tf.float64)
    location = tf.decode_raw(parsed['location_raw'], tf.int64)

    return image, location, label_one_hot
