"""
Step 1.
    Develop the training pipeline for all dataset. Generalized from Cifar 10 and previously Minc.
    As well as ImageNet.


Step 2
Use RunConfig to train model


"""
import sys

from keras.optimizers import SGD

from config2dataset import *
from config2model import *
from kyu.utils.example_engine import ExampleEnginev2, ExampleEngine

import keras.backend as K


def toggle_trainable_layers(model, trainable=True, keyword='', **kwargs):
    """
    Freeze the layers of a given model

    Parameters
    ----------
    model
    kwargs

    Returns
    -------
    model : keras.Model     need to be re-compiled once toggled.
    """
    for layer in model.layers:
        if keyword in layer.name:
            layer.trainable = trainable
    return model


def fit_model_v1(model, data,
                 load=False, save=True, verbose=1, title='default',
                 batch_size=32,
                 nb_epoch=200,
                 data_augmentation=False,
                 ):
    """
    General model fitting, given model (not compiled) and data. With some setting of parameters.

    Parameters
    ----------
    model : keras.model.Model   Keras model
    data : list                 [train, valid, test]
    load : bool
    save : bool
    verbose : int
    title : str

    Returns
    -------
    None
    """
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    save_log = True
    engine = ExampleEngine(data[0], model, data[1],
                           load_weight=load, save_weight=save, save_log=save_log,
                           lr_decay=True, early_stop=True, tensorboard=True,
                           batch_size=batch_size, nb_epoch=nb_epoch, title=title, verbose=verbose)

    if save_log:
        sys.stdout = engine.stdout
    model.summary()
    engine.fit(batch_size=batch_size, nb_epoch=nb_epoch, augmentation=data_augmentation)
    score = engine.model.evaluate(data[1][0], data[1][1], verbose=0)
    # engine.plot_result('loss')
    engine.plot_result()
    print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))
    if save_log:
        sys.stdout = engine.stdout.close()


def fit_model_v2(model, data,
                 load=False, save=True, verbose=1, title='default',
                 batch_size=32,
                 nb_epoch=200,
                 data_augmentation=False,
                 optimizer=None,
                 early_stop=False,
                 finetune=False,
                 ):
    """
    General model fitting, given model (not compiled) and data. With some setting of parameters.

    Parameters
    ----------
    model : keras.model.Model   Keras model
    data : list                 [train, valid, test]
    load : bool
    save : bool
    verbose : int
    title : str

    Returns
    -------
    None
    """
    if finetune:
        if optimizer is None:
            raise ValueError("fit_model_v2: Finetune must have a compiled model.")
    else:
        if optimizer is None:
            optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    save_log = True
    engine = ExampleEngine(data[0], model, data[1],
                           load_weight=load, save_weight=save, save_log=save_log,
                           lr_decay=True, early_stop=early_stop, tensorboard=True,
                           batch_size=batch_size, nb_epoch=nb_epoch, title=title, verbose=verbose)

    if save_log:
        sys.stdout = engine.stdout
    model.summary()
    engine.fit(batch_size=batch_size, nb_epoch=nb_epoch, augmentation=data_augmentation, verbose=verbose)
    # score = engine.model.evaluate(data[1][0], data[1][1], verbose=0)
    # engine.plot_result('loss')
    engine.plot_result()
    # print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))
    if save_log:
        sys.stdout = engine.stdout.close()


def run_resnet_o2(exp, config, image_gen):

    if config.mode == 'train':
        pass
    elif config.mode == 'finetune':
        nb_epoch_finetune = config.nb_epoch
        nb_epoch_after = config.nb_epoch_after
        cov_mode = config.cov_mode
        cov_branch = config.cov_branch
        cov_regularizer = config.cov_regularizer

        print("Running finetune for {}, experiment {}".format(config.title, exp))
        for param in config.params:
            for mode in config.mode_list:
                for cov_output in config.cov_outputs:
                    print("Run ResNet param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                sess = K.get_session()
                with sess.as_default():
                    model = ResNet50_o2(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                        nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape,
                                        last_avg=False,
                                        freeze_conv=True,
                                        cov_regularizer=cov_regularizer,
                                        last_conv_feature_maps=config.last_conv_feature_maps)
                    minc2500_finetune(model,
                                      title='minc2500_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                                      nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune,
                                      batch_size=config.batch_size, early_stop=early_stop, verbose=2,
                                      image_gen=image_gen)
                    model.save_weights(get_tmp_weights_path(model.name))

                K.clear_session()
                sess2 = K.get_session()
                with sess2.as_default():
                    model = ResNet50_o2(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                        nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape,
                                        last_avg=False,
                                        freeze_conv=False,
                                        cov_regularizer=cov_regularizer)
                    model.load_weights(get_tmp_weights_path(model.name))
                    minc2500_finetune(model,
                                      title='minc2500_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                                      nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after,
                                      batch_size=4, early_stop=early_stop, verbose=2)


# TODO Delay to March
def train_with_config(modelConfig, dataConfig, runConfig):
    """

    Parameters
    ----------
    runConfig

    Returns
    -------

    """
    # Create data sets
    data = config2data(dataConfig)

    # Create model
    model = config2model(modelConfig)

    # Create ExampleEngine v2
    engine = ExampleEnginev2(data, model, runConfig)
    engine.run()


