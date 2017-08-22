"""
Step 1.
    Develop the training pipeline for all dataset. Generalized from Cifar 10 and previously Minc.
    As well as ImageNet.


Step 2
Use RunConfig to train model


"""
import sys

from keras.optimizers import SGD
from kyu.engine.trainer import ClassificationTrainer
from kyu.models import get_model
from kyu.tensorflow.ops.math import StiefelSGD
from kyu.utils.example_engine import ExampleEngine
from kyu.utils.image import get_vgg_image_gen, get_resnet_image_gen


def fit_model_v1(model, data,
                 load=False, save=True, verbose=1, title='default',
                 batch_size=32,
                 nb_epoch=200,
                 data_augmentation=False,
                 lr=0.001,
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
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
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
    if nb_epoch > 0:
        engine.plot_result()
    print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))
    if save_log:
        sys.stdout = engine.stdout.close()


def fit_model_v2(model, data,
                 load=False, save=True, verbose=1, title='default',
                 batch_size=32,
                 nb_epoch=200,
                 data_augmentation=True,
                 optimizer=None,
                 early_stop=False,
                 finetune=False,
                 log=True,
                 lr=0.001,
                 lr_decay=True,
                 weight_path='',
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
            optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif isinstance(optimizer, StiefelSGD):
            """ Create the new optimizer to make sure it is in the same graph. """
            optimizer = StiefelSGD(lr, decay=optimizer.inital_decay, momentum=optimizer.init_momentum,
                                   observed_names=optimizer.observed_names,
                                   nesterov=optimizer.nesterov)
            print("----- Use lr = {} for StiefelSGD ----- ".format(lr))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

    save_log = log
    engine = ExampleEngine(data[0], model, data[1],
                           load_weight=load, save_weight=save, save_log=save_log,
                           lr_decay=lr_decay, early_stop=early_stop, tensorboard=True,
                           batch_size=batch_size, nb_epoch=nb_epoch, title=title, verbose=verbose,
                           weight_path=weight_path)

    if save_log:
        sys.stdout = engine.stdout
    model.summary()
    engine.fit(batch_size=batch_size, nb_epoch=nb_epoch, augmentation=data_augmentation, verbose=verbose)
    # score = engine.model.evaluate(data[1][0], data[1][1], verbose=0)
    # engine.plot_result('loss')
    if nb_epoch > 0:
        engine.plot_result()
    # print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))
    if save_log:
        sys.stdout = engine.stdout.close()


def finetune_with_model_data(data, model_config, dirhelper, nb_epoch_finetune, running_config):
    """
    Generic training pipeline provided with data, model_config and nb_epoch_finetune

    Parameters
    ----------
    data
    model_config
    nb_epoch_finetune
    running_config

    Returns
    -------

    """

    model_config.nb_class = data.nb_class
    if model_config.class_id == 'vgg':
        data.image_data_generator = get_vgg_image_gen(model_config.target_size,
                                                      running_config.rescale_small,
                                                      running_config.random_crop,
                                                      running_config.horizontal_flip)
    else:
        data.image_data_generator = get_resnet_image_gen(model_config.target_size,
                                                         running_config.rescale_small,
                                                         running_config.random_crop,
                                                         running_config.horizontal_flip)
    dirhelper.build(running_config.title)

    if nb_epoch_finetune > 0:
        # model_config2 = copy.copy(model_config)
        model_config.freeze_conv = True
        model = get_model(model_config)

        trainer = ClassificationTrainer(model, data, dirhelper,
                                        model_config=model_config, running_config=running_config,
                                        save_log=True,
                                        logfile=dirhelper.get_log_path())

        trainer.model.summary()
        trainer.fit(nb_epoch=nb_epoch_finetune, verbose=2)
        trainer.plot_result()
        # trainer.plot_model()
        model_config.freeze_conv = False
        running_config.load_weights = True
        running_config.init_weights_location = dirhelper.get_weight_path()

    model = get_model(model_config)

    trainer = ClassificationTrainer(model, data, dirhelper,
                                    model_config=model_config, running_config=running_config,
                                    save_log=True,
                                    logfile=dirhelper.get_log_path())

    trainer.build()

    trainer.fit(verbose=2)
    trainer.plot_result()


