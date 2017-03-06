"""
Finetune with DTD dataset
"""
import os
import warnings

import keras.backend as K
from keras.engine import merge
from keras.layers import Flatten, O2Transform, SecondaryStatistic

from keras.applications import ResNet50
from keras.layers import Dense
from kyu.models.keras_support import covariance_block_original, covariance_block_vector_space
from kyu.theano.general.config import DCovConfig
from kyu.theano.general.finetune import run_finetune, run_finetune_with_Stiefel_layer

os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from kyu.datasets.imagenet import preprocess_image_for_imagenet

from kyu.models.vgg import VGG16_o1, VGG16_o2
from kyu.models.resnet import ResNet50_o1, ResCovNet50, ResNet50_o2
from kyu.models.fitnet import fitnet_v1_o1, fitnet_v1_o2
from kyu.datasets.dtd import load_dtd
from kyu.theano.general.train import fit_model_v2, toggle_trainable_layers, Model

from kyu.models.minc import minc_fitnet_v2

import keras.backend as K
from keras.preprocessing.image import ImageDataGeneratorAdvanced, ImageDataGenerator

# Some constants
nb_classes = 47
if K.backend() == 'tensorflow':
    input_shape=(224,224,3)
    K.set_image_dim_ordering('tf')
else:
    input_shape=(3,224,224)
    K.set_image_dim_ordering('th')

TARGET_SIZE = (224,224)
RESCALE_SMALL = 256


RESNET_BASELINE_WEIGHTS_PATH = '/home/kyu/.keras/models/dtd_resnet50-baseline_resnet_1.weights'


def get_tmp_weights_path(name):
    return '/tmp/{}_finetune.weights'.format(name)


def model_finetune(base_model, pred_model, optimizer,
                   weights_path=RESNET_BASELINE_WEIGHTS_PATH,
                   loss='categorical_crossentropy', metric=['accuracy']):
    """
    Create a new model for fine-tune

    Parameters
    ----------
    base_model
    pred_model
    weights_path
    optimizer
    loss
    metric

    Returns
    -------

    """
    # Freeze the layers
    toggle_trainable_layers(base_model, False)
    base_model.load_weights(weights_path, by_name=True)
    new_model = Model(input=base_model.input, output=pred_model.output, name=base_model.name + "_" + pred_model.name)

    new_model.compile(optimizer, loss, metric)
    new_model.summary()
    return new_model, base_model, pred_model


def dtd_finetune(model,
                 nb_epoch_finetune=100, nb_epoch_after=0, batch_size=32,
                 image_gen=None,
                 title='dtd_finetune', early_stop=False,
                 keyword='',
                 log=True,
                 optimizer=None,
                 verbose=2,
                 lr=0.001):

    train, _, test = load_dtd(True, image_gen=image_gen, batch_size=batch_size)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    fit_model_v2(model, [train, test], batch_size=batch_size, title=title,
                 nb_epoch=nb_epoch_finetune,
                 optimizer=optimizer,
                 early_stop=early_stop,
                 log=log,
                 verbose=verbose,
                 lr=lr)
    tmp_weights = get_tmp_weights_path(model.name)
    model.save_weights(tmp_weights)
    if nb_epoch_after > 0:
        # K.clear_session()
        toggle_trainable_layers(model, True, keyword)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.load_weights(tmp_weights)
        fit_model_v2(model, [train, test], batch_size=batch_size, title=title,
                     nb_epoch=nb_epoch_after,
                     optimizer=optimizer,
                     early_stop=early_stop,
                     verbose=verbose,
                     lr=lr)

    return


def get_residual_cov_experiment(exp):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    if exp == 1:
        """ Test Multi branch ResNet 50 """
        nb_branch = 1
        params = [[257, 257, 257, 257, 257, 257],]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'residual'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
    elif exp == 2:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 2
        params = [[257, 257, 257],]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'residual'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp)
    return config


def get_von_settings(exp=1):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    cov_alpha = 0.01
    if exp == 1:
        """ Test Multi branch ResNet 50 """
        nb_branch = 1
        params = [[256, 128, 64],]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = None
        cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
        robust = True
        # cov_alpha = 0.75
    elif exp == 2:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 8
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'vN'
        last_config_feature_maps = []
        # last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        cov_alpha = 0.75
    elif exp == 3:
        """ Test Multi branch ResNet 50 """
        nb_branch = 1
        params = [[256, 128, 64], ]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        # cov_regularizer = None
        cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
        robust = False

    elif exp == 4:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 4
        params = [[257, 128, 64], [257, 128, 128, 64],]
        mode_list = [1]
        cov_outputs = [64, 32]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        cov_alpha = 0.75
    else:
        return

    if robust:
        rb = 'robost'
    else:
        rb = ''

    title = 'dtd_von_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha)
    return config


def get_von_with_regroup(exp=1):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    cov_alpha = 0.01
    if exp == 1:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 4
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 10
        robust = True
        regroup = True
        cov_alpha = 0.75
    elif exp == 2:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'pmean'
        cov_branch = 'o2transform'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        regroup = False
        cov_alpha = 0.75
    else:
        return

    if robust:
        rb = 'robost'
    else:
        rb = ''
    title = 'dtd_regroup_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup)
    return config

def get_constraints_settings(exp=1):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    cov_alpha = 0.01

    if exp == 1:
        """ Test Multi branch ResNet 50 """
        nb_branch = 1
        params = [[256, 128, 64], ]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [3]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = None
        cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
        robust = True
        # cov_alpha = 0.75
    else:
        return

    if robust:
        rb = 'robost'
    else:
        rb = ''

    title = 'dtd_cUN_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha)
    return config


def get_tensorboard_test_setting(exp=1):

    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    if exp == 1:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 4
        params = [[257, 128, 64], [257, 128, 128, 64],]
        mode_list = [1]
        cov_outputs = [64, 32]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        cov_alpha = 0.75
    else:
        return

    if robust:
        rb = 'robost'
    else:
        rb = ''

    title = 'dtd_von_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha)
    return config


def get_experiment_settings(exp=1):
    cov_regularizer = None
    nb_branch = 1
    batch_size = 32
    last_config_feature_maps = []
    if exp == 1:
        params = [[], [1024], [512], [1024, 512], [512, 256], [2048, 1444], [2048, 1024, 512]]
        mode_list = [1]
        cov_outputs = [512, 256, 128, 64]
        cov_branch = 'o2transform'
        cov_mode = 'channel'
        early_stop = True
    elif exp == 2:
        params = [[1024], [512], [1024, 512], [512, 256], [2048, 1444], [2048, 1024, 512]]
        mode_list = [1]
        cov_outputs = [512, 256, 128, 64]
        cov_branch = 'o2transform'
        cov_mode = 'mean'
        early_stop = True
    elif exp == 3:
        """Test the Regularizer """
        params = [[]]
        mode_list = [1]
        cov_outputs = [512, 256, 128]
        cov_mode = 'channel'
        cov_branch = 'o2transform'
        early_stop = True
        cov_regularizer = 'Fob'
    elif exp == 4:
        """Test VGG16 with DCov-2 """
        params = [[]]
        mode_list = [1]
        cov_outputs = [256]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
    elif exp == 5:
        """ Test LogTransform layers"""
        params = [[]]
        mode_list = [3]
        cov_outputs = [256]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
    elif exp == 6:
        """ Test ResNet 50 """
        params = [[256, 128]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = [1024, 512]
        # last_config_feature_maps = []

    elif exp == 7:
        """ Test ResNet 50 with different branches """
        nb_branch = 4
        params = [[256, 128]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'pmean'
        # cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        # last_config_feature_maps = []
        batch_size = 16

    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size)
    return config


def run_routine_resnet(config, verbose=(2,2), nb_epoch_finetune=10, nb_epoch_after=50,
                       stiefel_observed=None, stiefel_lr=0.01):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           )
    monitor_class = (O2Transform, SecondaryStatistic)
    # monitor_metrics = ['weight_norm',]
    # monitor_metrics = ['output_norm',]
    monitor_metrics = ['matrix_image',]
    if stiefel_observed is None:
        run_finetune(ResNet50_o2, dtd_finetune,
                     nb_classes=nb_classes,
                     input_shape=input_shape, config=config,
                     nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                     image_gen=image_gen, title='dtd_resnet50', verbose=verbose,
                     monitor_classes=monitor_class,
                     monitor_measures=monitor_metrics)
    else:
        run_finetune_with_Stiefel_layer(ResNet50_o2, dtd_finetune,
                                        nb_classes=nb_classes,
                                        input_shape=input_shape, config=config,
                                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                                        image_gen=image_gen, title='dtd_resnet50_stiefel', verbose=verbose,
                                        monitor_classes=monitor_class,
                                        monitor_measures=monitor_metrics,
                                        observed_keywords=stiefel_observed,
                                        lr=stiefel_lr)


def run_routine_vgg(exp=1):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    nb_epoch_finetune = 40
    nb_epoch_after = 50
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )
    config = get_experiment_settings(exp)
    cov_branch = config.cov_branch
    cov_mode = config.cov_mode
    cov_regularizer = config.cov_regularizer
    early_stop = config.early_stop

    print("Running experiment {}".format(exp))
    for param in config.params:
        for mode in config.mode_list:
            for cov_output in config.cov_outputs:
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))

                model = VGG16_o2(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                 nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape,
                                 freeze_conv=True, cov_regularizer=cov_regularizer)

                dtd_finetune(model, title='dtd_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                             nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                             image_gen=image_gen,
                             batch_size=32, early_stop=early_stop)


def baseline_finetune_vgg():
    nb_epoch_finetune = 10
    nb_epoch_after = 50

    model = VGG16_o1(denses=[4096,4096], nb_classes=nb_classes, input_shape=input_shape, freeze_conv=True)
    model.name = 'baseline_vgg16'
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )
    dtd_finetune(model, image_gen=image_gen, title='dtd_finetune_vgg16',
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 batch_size=32, early_stop=True)


def baseline_finetune_resnet(exp):
    nb_epoch_finetune = 10
    nb_epoch_after = 50
    if exp == 1:
        model = ResNet50_o1(denses=[], nb_classes=nb_classes, input_shape=input_shape, freeze_conv=True)
        model.name = 'baseline_resnet'
    elif exp == 2:
        last_conv_feature_maps = [1024, 512]
        model = ResNet50_o1(denses=[], nb_classes=nb_classes, input_shape=input_shape, freeze_conv=True,
                            last_conv_feature_maps=last_conv_feature_maps)
    else:
        return
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )
    dtd_finetune(model, image_gen=image_gen, title='dtd_finetune_vgg16',
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 batch_size=32, early_stop=True)


def test_routine_vgg(exp=1):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    nb_epoch_finetune = 10
    nb_epoch_after = 10

    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )
    config = get_experiment_settings(exp)
    cov_branch = config.cov_branch
    cov_mode = config.cov_mode
    cov_regularizer = config.cov_regularizer
    early_stop = config.early_stop

    print("Running experiment {}".format(exp))
    for param in config.params:
        for mode in config.mode_list:
            for cov_output in config.cov_outputs:
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))

                model = VGG16_o2(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                 nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape,
                                 freeze_conv=True, cov_regularizer=cov_regularizer)

                dtd_finetune(model, title='dtd_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                             nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                             image_gen=image_gen,
                             batch_size=32, early_stop=early_stop,
                             verbose=1)


def test_routine_resnet(config, verbose=(1,2), nb_epoch_finetune=1, nb_epoch_after=50):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           # channelwise_std_normalization=True
                                           )

    run_finetune(ResNet50_o2, dtd_finetune,
                 nb_classes=nb_classes,
                 input_shape=input_shape, config=config,
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 image_gen=image_gen, title='test_dtd_resnet', verbose=verbose)


def run_residual_cov_resnet(exp):
    """
        Finetune the ResNet-DCov

        Returns
        -------
    m
        """
    nb_epoch_finetune = 40
    nb_epoch_after = 50

    config = get_residual_cov_experiment(exp)
    cov_branch = config.cov_branch
    cov_mode = config.cov_mode
    cov_regularizer = config.cov_regularizer
    early_stop = config.early_stop

    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           # channelwise_std_normalization=True
                                           )

    print("Running experiment {}".format(exp))
    for param in config.params:
        for mode in config.mode_list:
            for cov_output in config.cov_outputs:
                sess = K.get_session()
                with sess.as_default():
                    title = 'dtd_residual_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode)
                    model = ResNet50_o2(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                        nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape,
                                        last_avg=False,
                                        freeze_conv=True,
                                        cov_regularizer=cov_regularizer,
                                        last_conv_feature_maps=config.last_conv_feature_maps,
                                        epsilon=config.epsilon)
                    dtd_finetune(model, title='finetune_' + title,
                                 nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune, log=True,
                                 batch_size=config.batch_size, early_stop=early_stop, verbose=2,
                                 image_gen=image_gen)

                    model.save_weights(get_tmp_weights_path(title + model.name))

                K.clear_session()
                sess2 = K.get_session()
                with sess2.as_default():
                    model = ResNet50_o2(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                        nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape,
                                        last_avg=False,
                                        freeze_conv=False,
                                        cov_regularizer=cov_regularizer,
                                        last_conv_feature_maps=config.last_conv_feature_maps,
                                        epsilon=config.epsilon
                                        )
                    model.load_weights(get_tmp_weights_path(title + model.name))
                    dtd_finetune(model, title='retrain_' + title,
                                 nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after, log=True,
                                 batch_size=config.batch_size / 4, early_stop=early_stop, verbose=2,
                                 image_gen=image_gen)



if __name__ == '__main__':
    # run_routine_resnet(6)
    # run_routine_vgg(4)
    # baseline_finetune_vgg()
    # baseline_finetune_resnet(2)
    # test_routine_vgg(6)
    # test_routine_resnet(7)

    # run_residual_cov_resnet(1)

    # config = get_von_settings(4)
    config = get_von_with_regroup(1)
    # config.batch_size = 4
    # config = get_constraints_settings(1)
    # config = get_experiment_settings(7)
    # config = get_experiment_settings()
    print(config.title)
    # test_routine_resnet(config, verbose=(2,1), nb_epoch_after=50, nb_epoch_finetune=2)
    # run_routine_resnet(config, verbose=(2,2), stiefel_observed=['o2t'], stiefel_lr=(0.01, 0.001), nb_epoch_finetune=10)
    run_routine_resnet(config, verbose=(1,2), stiefel_observed=['o2t'], stiefel_lr=(0.01, 0.001),
                       nb_epoch_finetune=1, nb_epoch_after=50)
