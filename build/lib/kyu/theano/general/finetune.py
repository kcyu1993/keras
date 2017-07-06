"""
Define general finetune process
"""
import os
import keras.backend as K
import numpy as np

import tensorflow as tf

from kyu.tensorflow.ops.math import get_matrix_norm, StiefelSGD
from kyu.theano.general.visualization import log_model
from third_party.openai.weightnorm import SGDWithWeightnorm


def get_tmp_weights_path(name):
    return '/home/kyu/.keras/models/tmp/{}_finetune.weights'.format(name)


def finetune_model_with_config(model, fn_function, config, nb_classes, input_shape,
                               title='default', image_gen=None,
                               verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                               stiefel_observed=None, stiefel_lr=0.01):

    # monitor_class = (O2Transform, SecondaryStatistic)
    # monitor_metrics = ['weight_norm',]
    # monitor_metrics = ['output_norm',]
    # monitor_metrics = ['matrix_image',]
    if stiefel_observed is None:
        run_finetune(model, fn_function,
                     nb_classes=nb_classes,
                     input_shape=input_shape, config=config,
                     nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                     image_gen=image_gen, title=title, verbose=verbose,
                     monitor_classes=[],
                     monitor_measures=[])
    else:
        run_finetune_with_Stiefel_layer(model, fn_function,
                                        nb_classes=nb_classes,
                                        input_shape=input_shape, config=config,
                                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                                        image_gen=image_gen, title=title + "-stiefel", verbose=verbose,
                                        monitor_classes=[],
                                        monitor_measures=[],
                                        observed_keywords=stiefel_observed,
                                        lr=stiefel_lr)


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


def run_finetune_with_weight_norm(
        fn_model, fn_finetune, input_shape, config,
        image_gen=None, nb_classes=0,
        nb_epoch_finetune=50, nb_epoch_after=50,
        title='', verbose=(2,2),
        monitor_measures=[],
        monitor_classes=[],
        observed_keywords=None,
        lr=(0.1, 0.01),
        lr_decay=True
        ):
    """
    Wrapper to Stiefel-layer SGD updated rule

    Parameters
    ----------
    fn_model
    fn_finetune
    input_shape
    config
    image_gen
    nb_classes
    nb_epoch_finetune
    nb_epoch_after
    title
    verbose
    monitor_measures
    monitor_classes
    observed_keywords
    lr

    Returns
    -------

    """
    # gsgd_0 = StiefelSGD(lr[0], 0.2, 0, False, observed_names=observed_keywords)
    # gsgd_1 = StiefelSGD(lr[1], 0.2, 0, False, observed_names=observed_keywords)
    gsgd_0 = SGDWithWeightnorm(lr[0], 0.2, 0, False)
    gsgd_1 = SGDWithWeightnorm(lr[0], 0.2, 0, False)
    run_finetune(fn_model, fn_finetune, input_shape, config,
                 image_gen=image_gen, nb_classes=nb_classes,
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 title=title, verbose=verbose,
                 monitor_measures=monitor_measures,
                 monitor_classes=monitor_classes,
                 optimizer=(gsgd_0, gsgd_1),
                 lr_decay=lr_decay
                 )


def run_finetune_with_Stiefel_layer(
        fn_model, fn_finetune, input_shape, config,
        image_gen=None, nb_classes=0,
        nb_epoch_finetune=50, nb_epoch_after=50,
        title='', verbose=(2,2),
        monitor_measures=[],
        monitor_classes=[],
        observed_keywords=None,
        lr=(0.1, 0.01),
        lr_decay=True
        ):
    """
    Wrapper to Stiefel-layer SGD updated rule

    Parameters
    ----------
    fn_model
    fn_finetune
    input_shape
    config
    image_gen
    nb_classes
    nb_epoch_finetune
    nb_epoch_after
    title
    verbose
    monitor_measures
    monitor_classes
    observed_keywords
    lr

    Returns
    -------

    """
    gsgd_0 = StiefelSGD(lr[0], 0.2, 0, False, observed_names=observed_keywords)
    gsgd_1 = StiefelSGD(lr[1], 0.2, 0, False, observed_names=observed_keywords)
    run_finetune(fn_model, fn_finetune, input_shape, config,
                 image_gen=image_gen, nb_classes=nb_classes,
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 title=title, verbose=verbose,
                 monitor_measures=monitor_measures,
                 monitor_classes=monitor_classes,
                 optimizer=(gsgd_0, gsgd_1),
                 lr_decay=lr_decay
                 )


def run_finetune(fn_model, fn_finetune, input_shape, config,
                 image_gen=None, nb_classes=0,
                 nb_epoch_finetune=50, nb_epoch_after=50,
                 title='', verbose=(2,2),
                 monitor_measures=[],
                 monitor_classes=[],
                 optimizer=None,
                 lr_decay=True):
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
    if optimizer is not None:
        opt1 = optimizer[0]
        opt2 = optimizer[1]
        lr1 = opt1.init_lr
        lr2 = opt2.init_lr
    else:
        opt1 = None
        opt2 = None
        lr1 = 0.01
        lr2 = 0.001
    kwargs = config.kwargs
    random_key = np.random.randint(1, 1000)



    print("Running experiment {}".format(config.exp))
    for param in config.params:
        for mode in config.mode_list:
            if len(config.cov_outputs) == 0:
                config.cov_outputs = [param[-1]]
            for cov_output in config.cov_outputs:
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                K.clear_session()
                sess = K.get_session()
                title = config.title + \
                        '_cov_{}_wv{}_{}'.format(config.cov_branch, str(cov_output), config.cov_mode)
                if nb_epoch_finetune > 0:
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
                                         load_weights=config.weight_path,
                                         **kwargs
                                         )
                        # Write config to file
                        config.to_configobj(folder=title, comments='finetune_{}-lr_{}-model_{}'.format(
                            title, lr1, model.name))
                        if monitor:
                            summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)

                        fn_finetune(model,
                                    title='finetune_' + title,
                                    nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune,
                                    batch_size=config.batch_size, early_stop=config.early_stop, verbose=verbose[0],
                                    image_gen=image_gen,
                                    optimizer=opt1,
                                    lr=lr1,
                                    weight_path=config.weight_path,
                                    lr_decay=lr_decay)
                        model.save_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))

                K.clear_session()
                sess2 = K.get_session()
                if nb_epoch_after > 0:
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
                        if nb_epoch_finetune > 0:
                            if os.path.exists(get_tmp_weights_path(model.name + '_' + str(random_key))):
                                model.load_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))
                            else:
                                print("ERROR !!!! NO TMP WEIGHTS FOUND")
                        config.to_configobj(folder=title, comments='retrain_{}-lr_{}-model_{}'.format(
                            title, lr1, model.name))
                        if monitor:
                            summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)
                        fn_finetune(model,
                                    title='retrain_' + title,
                                    nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after,
                                    batch_size=config.batch_size/4, early_stop=config.early_stop, verbose=verbose[1],
                                    image_gen=image_gen,
                                    optimizer=opt2,
                                    lr_decay=lr_decay,
                                    lr=lr2)
                    # Write config to file

"""
            if len(config.cov_outputs) == 0:
                cov_output = param[-1]
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                K.clear_session()
                sess = K.get_session()
                title = config.title + \
                        '_cov_{}_wv{}_{}'.format(config.cov_branch, str(cov_output), config.cov_mode)
                if nb_epoch_finetune > 0:
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
                                         load_weights=config.weight_path,
                                         **kwargs
                                         )
                        # Write config to file
                        config.to_configobj(folder=title, comments='finetune_{}-lr_{}-model_{}'.format(
                            title, lr1, model.name))
                        if monitor:
                            summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)

                        fn_finetune(model,
                                    title='finetune_' + title,
                                    nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune,
                                    batch_size=config.batch_size, early_stop=config.early_stop, verbose=verbose[0],
                                    image_gen=image_gen,
                                    optimizer=opt1,
                                    lr=lr1,
                                    weight_path=config.weight_path,
                                    lr_decay=lr_decay)
                        model.save_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))

                K.clear_session()
                sess2 = K.get_session()
                if nb_epoch_after > 0:
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
                        if nb_epoch_finetune > 0:
                            model.load_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))
                        config.to_configobj(folder=title, comments='retrain_{}-lr_{}-model_{}'.format(
                            title, lr1, model.name))
                        if monitor:
                            summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)
                        fn_finetune(model,
                                    title='retrain_' + title,
                                    nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after,
                                    batch_size=config.batch_size / 4, early_stop=config.early_stop, verbose=verbose[1],
                                    image_gen=image_gen,
                                    optimizer=opt2,
                                    lr_decay=lr_decay,
                                    lr=lr2)
                        # Write config to file
"""

def resume_finetune_with_Stiefel_layer(
        fn_model, fn_finetune, input_shape, config, weights_path,
        image_gen=None, nb_classes=0,
        nb_epoch_finetune=50, nb_epoch_after=50,
        title='', verbose=(2,2),
        monitor_measures=[],
        monitor_classes=[],
        observed_keywords=None,
        lr=(0.1, 0.01),
        by_name=False
        ):
    """
    Wrapper to Stiefel-layer SGD updated rule

    Parameters
    ----------
    fn_model
    fn_finetune
    input_shape
    config
    image_gen
    nb_classes
    nb_epoch_finetune
    nb_epoch_after
    title
    verbose
    monitor_measures
    monitor_classes
    observed_keywords
    lr

    Returns
    -------

    """
    gsgd_0 = StiefelSGD(lr[0], False, observed_names=observed_keywords)
    gsgd_1 = StiefelSGD(lr[1], False, observed_names=observed_keywords)
    resume_finetune(fn_model, fn_finetune, input_shape, config,
                    weights_path=weights_path,
                    image_gen=image_gen, nb_classes=nb_classes,
                    nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                    title=title, verbose=verbose,
                    monitor_measures=monitor_measures,
                    monitor_classes=monitor_classes,
                    optimizer=(gsgd_0, gsgd_1),
                    by_name=by_name
                    )


def resume_finetune(fn_model, fn_finetune, input_shape, config,
                    weights_path='',
                    image_gen=None, nb_classes=0,
                    nb_epoch_finetune=0, nb_epoch_after=50,
                    title='', verbose=(2,2),
                    monitor_measures=[],
                    monitor_classes=[],
                    optimizer=None,
                    by_name=False):
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
    if optimizer is not None:
        opt1 = optimizer[0]
        opt2 = optimizer[1]
        lr1 = opt1.init_lr
        lr2 = opt2.init_lr
    else:
        opt1 = None
        opt2 = None
        lr1 = 0.0001
        lr2 = 0.00001
    kwargs = config.kwargs
    random_key = np.random.randint(1, 1000)

    print("Resuming from previous training running experiment {}".format(config.exp))
    for param in config.params:
        for mode in config.mode_list:
            for cov_output in config.cov_outputs:
                print("Run routine param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                K.clear_session()
                sess = K.get_session()
                title = title + config.title + \
                        '_cov_{}_wv{}_{}'.format(config.cov_branch, str(cov_output), config.cov_mode)
                if nb_epoch_finetune > 0:
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
                        # Write config to file
                        config.to_configobj(folder=title, comments='finetune_{}-lr_{}-model_{}'.format(
                            title, lr1, model.name))
                        if monitor:
                            summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)
                        model.load_weights(weights_path, by_name=by_name)
                        fn_finetune(model,
                                    title='finetune_' + title,
                                    nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune,
                                    batch_size=config.batch_size, early_stop=config.early_stop, verbose=verbose[0],
                                    image_gen=image_gen,
                                    optimizer=opt1,
                                    lr=lr1)
                        weights_path = get_tmp_weights_path(model.name + '_' + str(random_key))
                        model.save_weights(weights_path)

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
                    model.load_weights(weights_path)
                    config.to_configobj(folder=title, comments='retrain_{}-lr_{}-model_{}'.format(
                        title, lr1, model.name))
                    if monitor:
                        summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)
                    fn_finetune(model,
                                title='retrain_' + title,
                                nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after,
                                batch_size=config.batch_size, early_stop=config.early_stop, verbose=verbose[1],
                                image_gen=image_gen,
                                optimizer=opt2,
                                lr=lr2)
                    # Write config to file


def log_model_to_path(fn_model, input_shape, config,
                 nb_classes=0,
                 title='', path=None
                 ):
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
    pesudo_input = np.random.randn(*([config.batch_size,] + list(input_shape)))
    pesudo_output = np.random.randn(config.batch_size, nb_classes)
    kwargs = config.kwargs
    print("Running experiment {}".format(config.exp))
    for param in config.params:
        for mode in config.mode_list:
            for cov_output in config.cov_outputs:
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                model_title = config.title + '_cov_{}_wv{}_{}'.format(
                    config.cov_branch, str(cov_output), config.cov_mode)
                with tf.device('/gpu:0'):
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
                    model.compile(optimizer='sgd', loss='categorical_crossentropy')

                    model.fit(pesudo_input,pesudo_output,
                              batch_size=config.batch_size,
                              nb_epoch=1,)
                writer = log_model(model, path=title + '/' + model_title)
