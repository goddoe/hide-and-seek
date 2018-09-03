import pprint
import os
import time
from multiprocessing.pool import Pool
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from models.tf_template import BaseTfClassifier
from models.inception.inception_v3 import inception_v3_base, inception_v3_arg_scope
from models.inception.inception_v1 import inception_v1_base, inception_v1_arg_scope
from models.alexnet.alexnet_v2 import alexnet_v2_base, alexnet_v2_arg_scope
from libs.various_utils import generate_id_with_date, get_date_time_prefix
from libs.image_utils import find_location_by_cam

slim = tf.contrib.slim

"""
CONSTANTS
"""
MODE_TRAIN_GLOBAL = 'MODE_TRAIN_GLOBAL'
MODE_TRAIN_ONLY_CLF = 'MODE_TRAIN_ONLY_CLF'
IMAGE_MEAN = 112.74660
IMAGE_PREPROCESSED_MEAN = -0.11571


def build_inception_v3_base(X, is_training, final_endpoint='Mixed_7c'):
    with slim.arg_scope(inception_v3_arg_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v3_base(X,
                                                final_endpoint=final_endpoint,
                                                scope='InceptionV3')
    return net, end_points, inception_v3_arg_scope


def build_inception_v1_base(X, is_training, final_endpoint='Mixed_5c'):
    with slim.arg_scope(inception_v1_arg_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v1_base(X,
                                                final_endpoint=final_endpoint,
                                                scope='InceptionV1')
    return net, end_points, inception_v1_arg_scope


def build_alexnet_v2_base(X, is_training, final_endpoint='conv5'):
    """
    end_points 
    'alexnet_v2/conv3'
    'alexnet_v2/conv4'
    'alexnet_v2/conv5'
    'alexnet_v2/pool5'
    """

    with slim.arg_scope(inception_v1_arg_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = alexnet_v2_base(X,
                                              final_endpoint=final_endpoint,
                                              scope='alexnet_v2')

    return net, end_points, alexnet_v2_arg_scope


build_model_base_dict = {
        'InceptionV3': build_inception_v3_base,
        'InceptionV1': build_inception_v3_base,
        'alexnet_v2': build_alexnet_v2_base,
        }
    
def build_cam(W, last_conv, Y, size, name=None):
    with tf.variable_scope(name or 'localization'):
        W_shape = W.get_shape().as_list()
        assert len(W_shape) == 4 or len(W_shape) == 2, "build cam W shape is wrong"

        if len(W_shape) == 4:
            W = tf.reshape(W, shape=W_shape[-2:])

        print("W shape :{}".format(W_shape))

        class_weight_by_row = tf.transpose(W)
        batch_class_weight = tf.gather(
            class_weight_by_row,
            tf.argmax(tf.cast(Y, tf.int32), axis=1))

        batch_class_weight = tf.reshape(
            batch_class_weight,
            shape=[-1, batch_class_weight.get_shape().as_list()[-1], 1])
        last_conv_resized = tf.image.resize_bilinear(
            last_conv, size)
        last_conv_flatten_w_h = tf.reshape(
            last_conv_resized,
            shape=[-1,
                   last_conv_resized.get_shape().as_list()[1] *
                   last_conv_resized.get_shape().as_list()[2],
                   last_conv_resized.get_shape().as_list()[3]])

        print("class_weight_by_row shape : {}".format(class_weight_by_row.get_shape().as_list()))
        print("batch_class_weight shape : {}".format(batch_class_weight.get_shape().as_list()))
        print("last_conv_flatten_w_h shape : {}".format(last_conv_flatten_w_h.get_shape().as_list()))

        cam = tf.reshape(tf.matmul(last_conv_flatten_w_h, batch_class_weight),
                         shape=[-1,
                                last_conv_resized.get_shape().as_list()[1],
                                last_conv_resized.get_shape().as_list()[2]], name='cam')
    return cam


class Detector(BaseTfClassifier):
    def __init__(self,
                 output_dim,
                 input_shape=None,
                 model_base_input_shape=(224, 224),
                 model_base_name="InceptionV3",
                 model_base_final_endpoint='Mixed_7c',
                 model_name="hide_and_seek",
                 optimizer=None,
                 cost_function=None,
                 flag_preprocess=False,
                 tensorboard_path=None,
                 device=None,
                 **kwargs):
        super().__init__()

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer

        if cost_function is None:
            cost_function = lambda Y_pred, Y: -tf.reduce_mean(Y * tf.log(Y_pred + 1e-12))

        self.model_name = model_name
        self.model_base_name = model_base_name

        self.mean = None
        self.std = None
        self.min_loss = None
        self.best_accuracy = None

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model_base_input_shape = model_base_input_shape
        self.model_base_final_endpoint = model_base_final_endpoint

        self.cost_function = cost_function
        self.optimizer = optimizer
        self.flag_preprocess = flag_preprocess
        self.tensorboard_path = tensorboard_path

        self.pool = None

        self.g = tf.Graph()
        with self.g.as_default():
            self.build_model()
            self.saver = tf.train.Saver()

        self.var_list = self.g.get_collection('variables')

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(graph=self.g,
                               config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.variables_initializer(self.var_list))

        self.pool = None


    def __del__(self):
        print("X"*30)
        print("X"*30)
        print("I'm dead")
        print("X"*30)
        print("X"*30)

    def build_model(self):
        with tf.variable_scope('input'):
            if self.input_shape:
                X = tf.placeholder(dtype=tf.float32,
                                   shape=[None, *self.input_shape],
                                   name="X")
            else:
                X = tf.placeholder(dtype=tf.float32,
                                   shape=[None, None, None, None])
            Y = tf.placeholder(dtype=tf.float32,
                               shape=[None, self.output_dim],
                               name="Y")

            learning_rate = tf.placeholder(
                dtype=tf.float32, name='learning_rate')

            reg_lambda = tf.placeholder_with_default(0., shape=None, name="reg_lambda")
            is_training = tf.placeholder_with_default(False, shape=None, name='is_training')
            dropout_keep_prob = tf.placeholder_with_default(1., shape=None, name="dropout_keep_prob")

            global_step = tf.Variable(0, trainable=False)


            X_preprocessed = self.preprocess(X)

            print("X_preprocessed shape: {}".format(X_preprocessed.get_shape().as_list()))

        net, end_points, model_arg_scope = build_model_base_dict[
            self.model_base_name](
                X_preprocessed,
                is_training,
                self.model_base_final_endpoint)

        with slim.arg_scope(model_arg_scope(weight_decay=reg_lambda)):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                with tf.variable_scope('clf'):
                    last_conv = slim.conv2d(
                        net, 1024, [3, 3], stride=1, padding='SAME', scope='conv')
                    gap = tf.reduce_mean(
                        last_conv, [1, 2], keep_dims=True, name='global_pool')
                    h = slim.dropout(gap, dropout_keep_prob, scope='dropout')
                    logits_before = slim.conv2d(
                        h, self.output_dim, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    logits = tf.squeeze(logits_before, [1, 2], name='spatial_squeeze')
                    Y_pred = slim.softmax(logits, scope='Predictions')

                    print("last_conv shape : {}".format(last_conv.get_shape().as_list()))
                    print("gap shape :{}".format(gap.get_shape().as_list()))
                    print("h shape :{}".format(h.get_shape().as_list()))
                    print("logits_before shape :{}".format(logits_before.get_shape().as_list()))
                    print("logits shape :{}".format(logits.get_shape().as_list()))

                W = tf.get_default_graph().get_tensor_by_name("clf/logits/weights:0")
                cam = build_cam(W=W, last_conv=last_conv, Y=Y, size=self.input_shape[:2],
                                name='localization')


        """
        loss
        """
        clf_scope_list = ['clf']
        clf_var_list = [clf_var
                        for scope in clf_scope_list
                        for clf_var in tf.contrib.framework.get_variables(scope)]
        clf_init = tf.variables_initializer(clf_var_list)
        print("clf_var_to_optimize_list")
        pprint.pprint(clf_var_list)

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

        optimizer_clf = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        updates_only_clf = self.optimize(cost=cost,
                                         optimizer=optimizer_clf,
                                         target_scope=None,
                                         var_list=clf_var_list,
                                         name='optimize_clf')

        updates = self.optimize(cost=cost,
                                optimizer=optimizer,
                                target_scope=None,
                                name='optimize_global')

        accuracy, correct_prediction = self.calc_metric(Y=Y,
                                                        Y_pred=Y_pred)

        self.update_dict = {
            MODE_TRAIN_GLOBAL: updates,
            MODE_TRAIN_ONLY_CLF: updates_only_clf,
            }



        """
        tensorboard
        """
        if self.tensorboard_path:
            self.valid_loss_ph = tf.placeholder(dtype=tf.float32,
                                                name="valid_loss_ph")
            self.train_loss_ph = tf.placeholder(dtype=tf.float32,
                                                name="train_loss_ph")
            self.valid_accuracy_ph = tf.placeholder(dtype=tf.float32,
                                                    name="valid_accuracy_ph")
            self.train_accuracy_ph = tf.placeholder(dtype=tf.float32,
                                                    name="train_accuracy_ph")

            self.valid_loss = tf.summary.scalar('valid_loss', self.valid_loss_ph)   
            self.train_loss = tf.summary.scalar('train_loss', self.train_loss_ph)

            self.valid_accuracy = tf.summary.scalar('valid_accuracy', self.valid_accuracy_ph)   
            self.train_accuracy = tf.summary.scalar('train_accuracy', self.train_accuracy_ph)

            self.merged_summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.tensorboard_path, graph=self.g)

        self.cam = cam
        self.X = X
        self.X_preprocessed = X_preprocessed
        self.Y = Y
        self.Y_pred = Y_pred
        self.correct_prediction = correct_prediction
        self.accuracy = accuracy
        self.cost = cost
        self.updates = updates
        self.updates_only_clf = updates_only_clf

        self.is_training = is_training
        self.learning_rate = learning_rate
        self.global_step = global_step
        self.reg_lambda = reg_lambda
        self.dropout_keep_prob = dropout_keep_prob

        self.clf_init = clf_init
        self.clf_var_list = clf_var_list


        self.g.add_to_collection('cam', cam)
        self.g.add_to_collection('Y_pred', Y_pred)
        self.g.add_to_collection('X', X)
        self.g.add_to_collection('X_preprocessed', X_preprocessed)
        self.g.add_to_collection('Y', Y)
        self.g.add_to_collection('accuracy', accuracy)
        self.g.add_to_collection('cost', cost)
        self.g.add_to_collection('is_training', is_training)


    def optimize(self, cost, optimizer, target_scope=None, var_list=None, name=None):
        with tf.variable_scope(name or 'calc_loss'):
            with tf.variable_scope('loss'):
                losses = []
                l2_regularizer = 0.
                if target_scope:
                    losses = [loss for loss in self.g.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if loss.name.startswith(tuple(target_scope))]
                else:
                    losses = self.g.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

                if losses:
                    l2_regularizer = tf.add_n(losses)

                grad_var_tuple_list = []
                clip = tf.constant(5.0, name='clip')

                for grad, var in optimizer.compute_gradients(cost + l2_regularizer, var_list=var_list):
                    if grad is None:
                        continue
                    if target_scope:
                        if not var.name.startswith(tuple(target_scope)):
                            continue
                    grad_var_tuple_list.append(
                        (tf.clip_by_value(grad, -clip, clip), var))
                updates = optimizer.apply_gradients(grad_var_tuple_list)

        return updates

    def calc_metric(self, Y, Y_pred, name=None):
        with tf.variable_scope(name or 'metric'):
            correct_prediction = tf.equal(
                tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        return accuracy, correct_prediction

    def location(self, X=None, Y=None, cam_list=None, thresh=0.5, batch_size=32):
        if cam_list is None:
            cam_list = self.calc_cam(X, Y, batch_size)

        pool = Pool()

        bbox_list = pool.starmap(
            find_location_by_cam, zip(cam_list,  [thresh]*len(cam_list)))
        pool.close()
        pool.join()
        return bbox_list

    def calc_cam(self, X, Y, batch_size=32):
        cam_list = []

        n_batch = len(X) // batch_size
        n_batch += 0 if len(X) % batch_size == 0 else 1
        for batch_i in range(n_batch):
            batch_x = X[batch_i *
                        batch_size: (batch_i + 1) * batch_size]
            batch_y = Y[batch_i *
                        batch_size: (batch_i + 1) * batch_size]

            batch_cam = self.sess.run(self.cam,
                                      feed_dict={self.X: batch_x,
                                                 self.Y: batch_y,
                                                 self.is_training: False})

            cam_list.append(batch_cam)
        cam_list = np.concatenate(cam_list, axis=0)
        return cam_list

    def preprocess(self, X):
        X = tf.image.resize_bilinear(X, self.model_base_input_shape[:2])
        X = tf.cast(X, tf.float32)
        X = tf.multiply(X, 1/255.)
        X = tf.subtract(X, 0.5)
        X = tf.multiply(X, 2.0)
        return X

    def check_accuracy_and_loss(self, X, Y, dataset_init_op, flag_preprocess=True):
        correct = 0
        total_num = 0
        loss = 0.

        if flag_preprocess:
            X_tensor = self.X
        else:
            X_tensor = self.X_preprocessed

        self.sess.run(dataset_init_op)
        while True:
            try:
                X_batch, Y_batch = self.sess.run([X, Y])
                correct_pred, batch_loss = self.sess.run(
                    [self.correct_prediction, self.cost],
                    feed_dict={
                        X_tensor: X_batch,
                        self.Y: Y_batch,
                        self.is_training: False})

                correct += correct_pred.sum()
                total_num += correct_pred.shape[0]
                loss += correct_pred.shape[0] * batch_loss
            except tf.errors.OutOfRangeError:
                break

        accuracy = float(correct) / total_num
        loss /= total_num
        return accuracy, loss, correct, total_num

    def train_with_dataset_api(
            self, X, Y, init_dataset_train, init_dataset_valid,
            n_epoch, learning_rate, reg_lambda, dropout_keep_prob, patience,
            mode=MODE_TRAIN_GLOBAL, flag_preprocess=False, verbose_interval=1, save_dir_path=None):

        try:
            if self.save_dir_path is None and save_dir_path is None:
                self.save_dir_path = "./tmp/{}".format(generate_id_with_date())

            if save_dir_path:
                self.save_dir_path = save_dir_path

            os.makedirs(self.save_dir_path)
        except Exception as e:
            print("*" * 30)
            print("Make directory with save_dir_path is failed")
            print("Maybe, there is directory already or error because of \"{}\"".format(str(e)))
        """
        Initialize
        """
        patience_origin = patience
        self.min_loss = 999999999.
        self.best_accuracy = 0.

        """
        Train only classifier
        """
        """
        tmp
        """

        if flag_preprocess:
            X_tensor = self.X
        else:
            X_tensor = self.X_preprocessed

        train_start_time = time.time()
        epoch_tqdm = tqdm(range(n_epoch))
        for epoch_i in epoch_tqdm:
            self.sess.run(init_dataset_train)
            batch_i = 0
            while True:
                try:
                    batch_start_time = time.time()
                    X_batch, Y_batch = self.sess.run([X, Y])

                    self.sess.run(
                        self.update_dict[mode],
                        feed_dict={X_tensor: X_batch,
                                   self.Y: Y_batch,
                                   self.learning_rate: learning_rate,
                                   self.reg_lambda: reg_lambda,
                                   self.dropout_keep_prob: dropout_keep_prob,
                                   self.is_training: True})

                    curr_time = time.time()
                    batch_time = curr_time - batch_start_time

                    epoch_tqdm.set_description(
                        "epoch {}, batch {} takes: {:0.2f} sec".format(
                            epoch_i, batch_i, batch_time))
                    batch_i += 1
                except tf.errors.OutOfRangeError:
                    break

            train_accuracy, train_loss, _, _ = self.check_accuracy_and_loss(
                X, Y, init_dataset_train, flag_preprocess=flag_preprocess)
            valid_accuracy, valid_loss, _, _ = self.check_accuracy_and_loss(
                X, Y, init_dataset_valid, flag_preprocess=flag_preprocess)

            self.report_dict['valid_loss'].append(valid_loss)
            self.report_dict['train_loss'].append(train_loss)
            self.report_dict['valid_accuracy'].append(valid_accuracy)
            self.report_dict['train_accuracy'].append(train_accuracy)
     
            if verbose_interval:
                if epoch_i % verbose_interval == 0:
                    print("-" * 30)
                    print("epoch_i : {}".format(epoch_i))
                    print("train loss: {}, train accuracy: {}".format(
                        train_loss, train_accuracy))
                    print("valid loss: {}, valid accuracy: {}".format(
                        valid_loss, valid_accuracy))
                    print("best valid loss: {}, best valid accuracy : {}".format(
                        self.min_loss, self.best_accuracy))

            if valid_accuracy > self.best_accuracy:
                patience = patience_origin
                self.min_loss = valid_loss
                self.best_accuracy = valid_accuracy

                meta = {
                            'input_dim': self.input_dim,
                            'output_dim': self.output_dim,
                            'min_loss': self.min_loss,
                            'best_accuracy': self.best_accuracy,
                            'mean': self.mean,
                            'std': self.std,
                            'flag_preprocess': self.flag_preprocess,
                        }
                self.meta.update(meta)
                self.best_ckpt_path = "{}/{}".format(self.save_dir_path,
                                                     self.model_name)
                self.best_ckpt_path = self.save(self.best_ckpt_path)

                print("*" * 30)
                print("epoh_i : {}".format(epoch_i))
                print("train loss: {}, train accuracy: {}".format(
                    train_loss, train_accuracy))
                print("valid loss: {}, valid accuracy: {}".format(
                    valid_loss, valid_accuracy))
                print("best valid loss: {}, best valid accuracy : {}".format(
                    self.min_loss, self.best_accuracy))
                print("save current model : {}".format(self.best_ckpt_path))
            else:
                patience -= 1
            if patience <= 0:
                break

        print("train takes : {} sec".format(time.time() - train_start_time))
        self.load(self.best_ckpt_path)

        train_accuracy, train_loss, _, _ = self.check_accuracy_and_loss(
            X, Y, init_dataset_train, flag_preprocess=flag_preprocess)
        valid_accuracy, valid_loss, _, _ = self.check_accuracy_and_loss(
            X, Y, init_dataset_valid, flag_preprocess=flag_preprocess)

        self.meta['report_dict'] = self.report_dict

        date_time_prefix = get_date_time_prefix()
        self.final_model_path = "{}/{}_final_{}".format(
            self.save_dir_path, date_time_prefix, self.model_name)

        self.save(self.final_model_path)
        print("*"*30)
        print("final trained performance")
        print("train loss: {}, train accuracy: {}".format(
                    train_loss, train_accuracy))
        print("valid loss: {}, valid accuracy: {}".format(
                    valid_loss, valid_accuracy))
        print("best valid loss: {}, best valid accuracy : {}".format(
                    self.min_loss, self.best_accuracy))
        print("final_model_path: {}".format(self.final_model_path))
        print("train done")
        print("*"*30)

        return self

