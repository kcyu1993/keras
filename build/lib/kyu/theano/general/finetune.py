"""
Define general finetune process
"""

import keras.backend as K
import numpy as np

import tensorflow as tf

from kyu.tensorflow.ops.math import get_matrix_norm


def get_tmp_weights_path(name):
    return '/tmp/{}_finetune.weights'.format(name)


def create_summary_op_for_keras_model(model, layers_class, measures=['output_norm']):
    """
    Create TensorFlow

    Parameters
    ----------
    model
    layers_class
    measures

    Returns
    -------

    """
    if layers_class is None or len(layers_class) == 0:
        return

    if 'weight_norm' in measures:
        with tf.name_scope('weight_norm'):
            layers = model.layers
            for layer in layers:
                if isinstance(layer, layers_class):
                    with tf.name_scope(layer.name):
                        for weight in layer.weights:
                            tf.summary.scalar( weight.name + '.norm', get_matrix_norm(weight))

    if 'output_norm' in measures:
        with tf.name_scope('weight_norm'):
            layers = model.layers
            for layer in layers:
                if isinstance(layer, layers_class):
                    with tf.name_scope(layer.name):
                        output = layer.output
                        tf.summary.tensor_summary('output_norm', get_matrix_norm(output, True))

    if 'matrix_image' in measures:
        with tf.name_scope('matrix_image'):
            layers = model.layers
            for layer in layers:
                if isinstance(layer, layers_class):
                    with tf.name_scope(layer.name):
                        output = layer.output
                        tf.summary.image('output', output, max_outputs=2)

    merged_summary_op = tf.summary.merge_all()
    return merged_summary_op


def run_finetune(fn_model, fn_finetune, input_shape, config,
                 image_gen=None, nb_classes=0,
                 nb_epoch_finetune=50, nb_epoch_after=50,
                 title='', verbose=(2,2),
                 monitor_measures=[],
                 monitor_classes=[]):
    """
    General routine for finetune running.

    Should deal with the temporal saving and loading process.

    Parameters
    ----------
    fn_model : function to get the model, with interface defined.
    fn_finetune
    input_shape
    config
    nb_epoch_finetune
    nb_epoch_after
    image_gen
    nb_classes
    title

    Returns
    -------

    """
    monitor = False
    if len(monitor_classes) > 0 and len(monitor_classes) > 0:
        monitor = True

    kwargs = config.kwargs
    random_key = np.random.randint(1, 1000)
    print("Running experiment {}".format(config.exp))
    for param in config.params:
        for mode in config.mode_list:
            for cov_output in config.cov_outputs:
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                sess = K.get_session()
                title = title + config.title + \
                        '_cov_{}_wv{}_{}'.format(config.cov_branch, str(cov_output), config.cov_mode)
                with sess.as_default():
                    model = fn_model(parametrics=param, mode=mode, cov_branch=config.cov_branch,
                                     cov_mode=config.cov_mode,
                                     nb_classes=nb_classes, cov_branch_output=cov_output,
                                     input_shape=input_shape,
                                     last_avg=False,
                                     freeze_conv=True,
                                     cov_regularizer=config.cov_regularizer,
                                     nb_branch=config.nb_branch,
                                     last_conv_feature_maps=config.last_conv_feature_maps,
                                     epsilon=config.epsilon,
                                     vectorization=config.vectorization,
                                     **kwargs
                                     )
                    if monitor:
                        summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)

                    fn_finetune(model,
                                title='finetune_' + title,
                                nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune,
                                batch_size=config.batch_size, early_stop=config.early_stop, verbose=verbose[0],
                                image_gen=image_gen,
                                lr=1e-3)
                    model.save_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))

                K.clear_session()
                sess2 = K.get_session()
                with sess2.as_default():
                    model = fn_model(parametrics=param, mode=mode, cov_branch=config.cov_branch,
                                     cov_mode=config.cov_mode,
                                     nb_classes=nb_classes, cov_branch_output=cov_output,
                                     input_shape=input_shape,
                                     last_avg=False,
                                     freeze_conv=False,
                                     cov_regularizer=config.cov_regularizer,
                                     nb_branch=config.nb_branch,
                                     last_conv_feature_maps=config.last_conv_feature_maps,
                                     epsilon=config.epsilon,
                                     vectorization=config.vectorization,
                                     **kwargs
                                     )
                    model.load_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))
                    if monitor:
                        summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)
                    fn_finetune(model,
                                title='retrain_' + title,
                                nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after,
                                batch_size=config.batch_size/2, early_stop=config.early_stop, verbose=verbose[1],
                                image_gen=image_gen,
                                lr=1e-4)


