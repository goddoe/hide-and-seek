import os
import pickle
import inspect
from types import MethodType

import tensorflow as tf
import numpy as np

from libs.various_utils import generate_id_with_date, get_date_time_prefix

class BaseTfClassifier(object):
    def __init__(self,**kwargs):
        """
        Implement things below 
        """
        self.input_dim = None
        self.output_dim = None

        self.mean = None
        self.std = None
        self.min_loss = None

        self.best_accuracy = None
        self.flag_preprocess = None

        self.sess = None
        self.saver = None

        self.meta = None
        self.tensorboard_path = None
        self.save_dir_path = None
        self.model_name = 'model'
        self.meta = {}
        self.report_dict = {
                'valid_loss': [],
                'train_loss': [],
                'valid_accuracy': [],
                'train_accuracy': [],
                }
        self.g = None
        """
        self.build_model()
        """

    def build_model(self):
        """
        Implement things below 
        """
        self.X = None
        self.Y = None
        self.Y_pred = None
        self.accuracy = None
        self.updates = None

        self.learning_rate = None
        self.reg_lambda = None
        self.is_training = None
        self.cost = None
        self.optimizer = None


    def load(self, path, flag_import_graph=False, model=None):
        meta_path = "{}.meta".format(path)

        if flag_import_graph:
            self.g = tf.Graph()
            with self.g.as_default():
                self.sess = tf.Session(graph=self.g)
                self.saver = tf.train.import_meta_graph("{}.graph".format(path))
                self.saver.restore(self.sess, path)
                
                for key in self.g.get_all_collection_keys():
                    setattr(self, key, self.g.get_collection(key)[0])

        else:
            self.saver.restore(self.sess, path)

        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
            
        for key,val in self.meta.items():
            setattr(self, key, val)

        if model:
            func_tuple_list = inspect.getmembers(model, predicate=inspect.isfunction)
            for func_name, func in func_tuple_list:
                setattr(self, func_name, MethodType(func, self)) 
        return self

    def save(self,
             path,
             global_step=None,
             meta_graph_suffix='graph',
             latest_filename=None,
             write_meta_graph=True,
             write_state=True,
             flag_saved_model=False):

        if not os.path.exists(os.path.dirname(path)):
            try: os.makedirs(os.path.dirname(path))
            except Exception as e: print(e)

        meta_path = "{}.meta".format(path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta, f)

        return self.saver.save(self.sess,
                               path,
                               global_step=global_step,
                               latest_filename=latest_filename,
                               meta_graph_suffix=meta_graph_suffix,
                               write_meta_graph=write_meta_graph,
                               write_state=True)

    def save_with_saved_model(self, path):
        """
        Saved model
        """
        try:
            self.builder = tf.saved_model.builder.SavedModelBuilder(path)
            tensor_info_X = tf.saved_model.utils.build_tensor_info(self.X)
            tensor_info_Y = tf.saved_model.utils.build_tensor_info(self.Y_pred)
            tensor_info_istraining = tf.saved_model.utils.build_tensor_info(
                self.is_training)
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'X': tensor_info_X,
                            'is_training': tensor_info_istraining},
                    outputs={'Y_pred': tensor_info_Y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            init = tf.variables_initializer(self.g.get_collection('variables'))
            legacy_init_op = tf.group(
                init, name='legacy_init_op')

            self.builder.add_meta_graph_and_variables(
                self.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={'predict': prediction_signature},
                legacy_init_op=legacy_init_op)
            self.builder.save()
        except Exception as e:
            raise Exception("error in save_with_saved_model : {}".format(str(e)))

        return path

    def predict(self, X_target, batch_size=32):
        Y_pred_list = []

        if self.flag_preprocess:
            X_target = X_target.copy()
            X_target = self.preprocess(X_target)

        n_batch = len(X_target) // batch_size
        n_batch += 0 if len(X_target) % batch_size == 0 else 1
        for batch_i in range(n_batch):
            batch_x = X_target[batch_i *
                               batch_size: (batch_i + 1) * batch_size]
            batch_Y_pred = self.sess.run(self.Y_pred, 
                                    feed_dict={self.X: batch_x, 
                                                self.is_training: False})

            Y_pred_list.append(batch_Y_pred)
        Y_pred = np.concatenate(Y_pred_list, axis=0)
        return Y_pred

    def evaluate(self, X_target, Y_target, batch_size=32):
        if self.flag_preprocess:
            X_target = X_target.copy()
            X_target = self.preprocess(X_target)

        Y_pred_list = []
        accuracy = 0.
        loss = 0.

        n_batch = len(X_target) // batch_size
        n_batch += 0 if len(X_target) % batch_size == 0 else 1
        for batch_i in range(n_batch):
            batch_x = X_target[batch_i *
                               batch_size: (batch_i + 1) * batch_size]
            batch_y = Y_target[batch_i *
                               batch_size: (batch_i + 1) * batch_size]

            batch_Y_pred, batch_accuracy, batch_loss = self.sess.run([self.Y_pred, self.accuracy, self.cost],
                                                                     feed_dict={self.X: batch_x, 
                                                                                self.Y: batch_y,
                                                                                self.is_training: False})
            accuracy += len(batch_x) * batch_accuracy
            loss += len(batch_y) * batch_loss
            Y_pred_list.append(batch_Y_pred)

        Y_pred = np.concatenate(Y_pred_list, axis=0)
        accuracy /= len(X_target)
        loss /= len(X_target)

        return Y_pred, accuracy, loss

    def calc_moments_of_data(self, X_target=None):
        mean = np.mean(X_target, axis=0)
        std = np.std(X_target, axis=0)
        return mean, std

    def preprocess(self, X_target):
        return (X_target - self.mean) / (self.std + 1e-3)

    def prepare_preprocess(self, X_target):
        self.mean, self.std = self.calc_moments_of_data(X_target)

    def train(self, X_train, Y_train, X_valid, Y_valid, batch_size, n_epoch, learning_rate, reg_lambda=0., patience=100, verbose_interval=20, save_dir_path=None, **kwargs):

        try:
            if self.save_dir_path  is None and save_dir_path is None:
                self.save_dir_path = "./tmp/{}".format(generate_id_with_date())

            if save_dir_path:
                self.save_dir_path = save_dir_path

            os.makedirs(self.save_dir_path)
        except Exception as e:
            print("*" * 30)
            print("Make directory with save_dir_path is failed")
            print("Maybe, there is directory already or error because of \"{}\"".format(str(e)))

        X_train_org = X_train
        if self.flag_preprocess:
            print("-" * 30)
            print("preprocess start")
            self.prepare_preprocess(X_train)
            X_train = self.preprocess(X_train)
            print("preprocess done")

        print("-" * 30)
        print("train start")
        patience_origin = patience
        if self.min_loss is None:
            self.min_loss = 999999999.
        for epoch_i in range(n_epoch):
            rand_idx_list = np.random.permutation(range(len(X_train)))
            n_batch = len(rand_idx_list) // batch_size
            for batch_i in range(n_batch):
                rand_idx = rand_idx_list[batch_i *
                                         batch_size: (batch_i + 1) * batch_size]
                batch_x = X_train[rand_idx]
                batch_y = Y_train[rand_idx]
                
                self.sess.run(self.updates,
                                  feed_dict={self.X: batch_x,
                                             self.Y: batch_y,
                                             self.learning_rate: learning_rate,
                                             self.reg_lambda: reg_lambda,
                                             self.is_training: True})

            _, valid_accuracy, valid_loss = self.evaluate(
                X_valid, Y_valid, batch_size)
            _, train_accuracy, train_loss = self.evaluate(
                X_train_org, Y_train, batch_size)

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

            if valid_loss < self.min_loss:
                patience = patience_origin + 1

                self.min_loss = valid_loss
                self.best_accuracy = valid_accuracy

                meta = {
                            'input_dim':self.input_dim,
                            'output_dim':self.output_dim,
                            'min_loss':self.min_loss,
                            'best_accuracy':self.best_accuracy,
                            'mean':self.mean,
                            'std':self.std,
                            'flag_preprocess':self.flag_preprocess,
                        }
                self.meta.update(meta)
                self.save_path = "{}/{}".format(self.save_dir_path, self.model_name)
                self.best_ckpt_path = self.save(self.save_path)

                print("*" * 30)
                print("epoh_i : {}".format(epoch_i))
                print("train loss: {}, train accuracy: {}".format(
                    train_loss, train_accuracy))
                print("valid loss: {}, valid accuracy: {}".format(
                    valid_loss, valid_accuracy))
                print("best valid loss: {}, best valid accuracy : {}".format(
                    self.min_loss, self.best_accuracy))
                print("save current model : {}".format(self.best_ckpt_path))

            patience -= 1
            if patience <= 0:
                break

        self.load(self.best_ckpt_path)
        _, valid_accuracy, valid_loss = self.evaluate(X_valid, Y_valid, batch_size)
        _, train_accuracy, train_loss = self.evaluate(X_train_org, Y_train, batch_size)
        self.meta['report_dict'] = self.report_dict

        date_time_prefix = get_date_time_prefix()
        self.final_model_path = "{}/{}_final_{}".format(self.save_dir_path, date_time_prefix, self.model_name) 
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

        return self.sess
