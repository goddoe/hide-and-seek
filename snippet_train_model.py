import pickle 
from importlib import reload

import tensorflow as tf

import models.custom.detector as g
from libs.dataset_utils import prepare_data_from_tfrecord
from configs.project_config import project_path


"""
Constant
"""
NUM_CLASS = 200
INPUT_SHAPE = (64, 64, 3)
MODEL_BASE_INPUT_SHAPE = (224, 224, 3)

tfrecord_train_dir = "{}/data/tiny-imagenet-200/tfrecord/train".format(project_path)
tfrecord_valid_dir = "{}/data/tiny-imagenet-200/tfrecord/valid".format(project_path)
tfrecord_test_dir = "{}/data/tiny-imagenet-200/tfrecord/test".format(project_path)

meta_path = "{}/data/tiny-imagenet-200/meta.pickle".format(project_path)
pretrained_ckpt_path = "{}/checkpoints/inception_v3/inception_v3.ckpt".format(project_path)

vanila_model_save_path = "{}/checkpoints/vanila_inception_v3/vanila_inception_v3".format(project_path)
has_model_save_path = "{}/checkpoints/has_inception_v3/has_inception_v3".format(project_path)

# ==============================================================================

model_base_name = "alexnet_v2"
model = g.Detector(output_dim=NUM_CLASS,
                   input_shape=INPUT_SHAPE,
                   model_base_input_shape=MODEL_BASE_INPUT_SHAPE,
                   model_base_name=model_base_name,
                   model_name="hide_and_seek")

# ==============================================================================
with model.g.as_default():
    """
    Read Data
    """
    d = prepare_data_from_tfrecord(
        tfrecord_train_dir=tfrecord_train_dir,
        tfrecord_valid_dir=tfrecord_valid_dir,
        tfrecord_test_dir=tfrecord_test_dir,
        batch_size=200)
    (X, Y,
     init_dataset_train,
     init_dataset_train_has,
     init_dataset_valid) = (d['X'], d['Y'],
                            d['init_dataset_train'],
                            d['init_dataset_train_has'],
                            d['init_dataset_valid'])

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    model.meta.update(meta)

    """
    Initialize with pretrained weights
    """
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
        include=[model_base_name])
    init_pretrain_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        pretrained_ckpt_path, variables_to_restore)

    init_pretrain_fn(model.sess)


# ==============================================================================
"""
Vanila
"""

model.train_with_dataset_api(X=X,
                             Y=Y,
                             init_dataset_train=init_dataset_train,
                             init_dataset_valid=init_dataset_valid,
                             n_epoch=10,
                             learning_rate=0.001,
                             reg_lambda=0.,
                             dropout_keep_prob=8.,
                             patience=10,
                             verbose_interval=1,
                             mode=g.MODE_TRAIN_ONLY_CLF,
                             save_dir_path=None)


model.train_with_dataset_api(X=X,
                             Y=Y,
                             init_dataset_train=init_dataset_train,
                             init_dataset_valid=init_dataset_valid,
                             n_epoch=3,
                             learning_rate=0.001,
                             reg_lambda=0.,
                             dropout_keep_prob=8.,
                             patience=10,
                             verbose_interval=1,
                             mode=g.MODE_TRAIN_GLOBAL,
                             save_dir_path=None)

model.save(vanila_model_save_path)


# ==============================================================================
"""
HaS
"""

model.train_with_dataset_api(X=X,
                             Y=Y,
                             init_dataset_train=init_dataset_train_has,
                             init_dataset_valid=init_dataset_valid,
                             n_epoch=10,
                             learning_rate=0.001,
                             reg_lambda=0.,
                             dropout_keep_prob=8.,
                             patience=10,
                             verbose_interval=1,
                             mode=g.MODE_TRAIN_ONLY_CLF,
                             flag_has=True,
                             save_dir_path=None)


model.train_with_dataset_api(X=X,
                             Y=Y,
                             init_dataset_train=init_dataset_train_has,
                             init_dataset_valid=init_dataset_valid,
                             n_epoch=3,
                             learning_rate=0.001,
                             reg_lambda=0.,
                             dropout_keep_prob=8.,
                             patience=10,
                             verbose_interval=1,
                             mode=g.MODE_TRAIN_GLOBAL,
                             flag_has=True,
                             save_dir_path=None)

model.save(has_model_save_path)
