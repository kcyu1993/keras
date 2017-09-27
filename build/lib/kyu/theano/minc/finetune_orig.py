"""
Finetune with MINC dataset
"""
import os
import warnings

import keras.backend as K
from keras.engine import merge
from keras.layers import Flatten, SecondaryStatistic, O2Transform

from keras.applications import ResNet50
from keras.layers import Dense
from kyu.models.so_cnn_helper import covariance_block_original
from kyu.legacy.so_cnn_helper import covariance_block_vector_space
from kyu.theano.general.config import DCovConfig
from kyu.theano.general.finetune import run_finetune, run_finetune_with_Stiefel_layer, log_model_to_path

os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from kyu.utils.imagenet_utils import preprocess_image_for_imagenet

from kyu.legacy.vgg16 import VGG16_o1, VGG16_o2
from kyu.legacy.resnet50 import ResNet50_o1, ResNet50_o2_multibranch, ResNet50_o2, ResCovNet50

from kyu.datasets.minc import Minc2500, load_minc2500, MincOriginal

from kyu.theano.general.train import fit_model_v2, Model
from kyu.utils.train_utils import toggle_trainable_layers

import keras.backend as K
from keras.preprocessing.image import ImageDataGeneratorAdvanced, ImageDataGenerator
from kyu.theano.minc.configs import get_experiment_settings, get_aaai_experiment, get_log_experiment, get_von_settings, \
    get_von_with_regroup, get_von_with_multibranch, get_residual_cov_experiment, get_VGG_testing_ideas

# Some constants
nb_classes = 23
if K.backend() == 'tensorflow':
    input_shape=(224,224,3)
    K.set_image_dim_ordering('tf')
else:
    input_shape=(3,224,224)
    K.set_image_dim_ordering('th')

TARGET_SIZE = (224,224)
RESCALE_SMALL = 256


RESNET_BASELINE_WEIGHTS_PATH = '/home/kyu/.keras/models/MINC2500_resnet50-baseline_resnet_1.weights'


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


def mincorig_finetune(model,
                      nb_epoch_finetune=100, nb_epoch_after=0, batch_size=32,
                      image_gen=None,
                      title='MINCorig_finetune',
                      early_stop=False,
                      keyword='',
                      optimizer=None,
                      weight_path='',
                      log=True,
                      verbose=2,
                      load=False,
                      lr_decay=True,
                      lr=0.001):


    loader = MincOriginal()
    train = loader.generator('train.txt', batch_size=batch_size, gen=image_gen, target_size=TARGET_SIZE)
    test = loader.generator('validate.txt', batch_size=batch_size, gen=image_gen, target_size=TARGET_SIZE)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    fit_model_v2(model, [train, test], batch_size=batch_size, title=title,
                 nb_epoch=nb_epoch_finetune,
                 optimizer=optimizer,
                 early_stop=early_stop,
                 verbose=verbose,
                 weight_path=weight_path,
                 lr_decay=lr_decay,
                 log=log,
                 lr=lr)
    tmp_weights = get_tmp_weights_path(model.name)
    if nb_epoch_finetune > 0:
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
                     lr=lr/10)

    return


def run_routine_resnet(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
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
        run_finetune(ResNet50_o2, mincorig_finetune,
                     nb_classes=nb_classes,
                     input_shape=input_shape, config=config,
                     nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                     image_gen=image_gen, title='minc_orig_resnet50', verbose=verbose,
                     monitor_classes=monitor_class,
                     monitor_measures=monitor_metrics)
    else:
        run_finetune_with_Stiefel_layer(ResNet50_o2, mincorig_finetune,
                                        nb_classes=nb_classes,
                                        input_shape=input_shape, config=config,
                                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                                        image_gen=image_gen, title='minc_orig_resnet50_stiefel', verbose=verbose,
                                        monitor_classes=monitor_class,
                                        monitor_measures=monitor_metrics,
                                        observed_keywords=stiefel_observed,
                                        lr=stiefel_lr)


def run_routine_vgg(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                    stiefel_observed=None, stiefel_lr=0.01):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           preprocessing_function=preprocess_image_for_imagenet
                                           )

    monitor_class = (O2Transform, SecondaryStatistic)
    # monitor_metrics = ['weight_norm',]
    # monitor_metrics = ['output_norm',]
    monitor_metrics = ['matrix_image',]
    if stiefel_observed is None:
        run_finetune(VGG16_o2, mincorig_finetune,
                     nb_classes=nb_classes,
                     input_shape=input_shape, config=config,
                     nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                     image_gen=image_gen, title='mincorig_VGG', verbose=verbose,
                     monitor_classes=monitor_class,
                     monitor_measures=monitor_metrics)
    else:
        run_finetune_with_Stiefel_layer(VGG16_o2, mincorig_finetune,
                                        nb_classes=nb_classes,
                                        input_shape=input_shape, config=config,
                                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                                        image_gen=image_gen, title='minc_orig_stiefel', verbose=verbose,
                                        monitor_classes=monitor_class,
                                        monitor_measures=monitor_metrics,
                                        observed_keywords=stiefel_observed,
                                        lr=stiefel_lr)


def run_routine_resnet_multibranch(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                                   stiefel_observed=None, stiefel_lr=0.01):
    """
        Finetune the ResNet-DCov

        Returns
        -------

        """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           preprocessing_function=preprocess_image_for_imagenet
                                           )

    run_model_with_config(ResNet50_o2_multibranch, config, title='minc_orig_resnet_MB',
                          verbose=verbose, image_gen=image_gen,
                          nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                          stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed)


def run_model_with_config(model, config, title='minc_orig',
                          image_gen=None,
                          verbose=(2, 2), nb_epoch_finetune=15, nb_epoch_after=50,
                          stiefel_observed=None, stiefel_lr=0.01):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """

    if image_gen is None:
        image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                               horizontal_flip=True,
                                               )

    monitor_class = (O2Transform, SecondaryStatistic)
    # monitor_metrics = ['weight_norm',]
    # monitor_metrics = ['output_norm',]
    monitor_metrics = ['matrix_image',]
    if stiefel_observed is None:
        run_finetune(model, mincorig_finetune,
                     nb_classes=nb_classes,
                     input_shape=input_shape, config=config,
                     nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                     image_gen=image_gen, title=title, verbose=verbose,
                     monitor_classes=monitor_class,
                     monitor_measures=monitor_metrics)
    else:
        run_finetune_with_Stiefel_layer(model, mincorig_finetune,
                                        nb_classes=nb_classes,
                                        input_shape=input_shape, config=config,
                                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                                        image_gen=image_gen, title=title + "-stiefel", verbose=verbose,
                                        monitor_classes=monitor_class,
                                        monitor_measures=monitor_metrics,
                                        observed_keywords=stiefel_observed,
                                        lr=stiefel_lr)



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
    mincorig_finetune(model, image_gen=image_gen, title='mincorig_finetune_vgg16',
                      nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                      batch_size=32, early_stop=True)


def run_residual_cov_resnet(exp):
    """
        Finetune the ResNet-DCov

        Returns
        -------
    m
        """
    nb_epoch_finetune = 50
    nb_epoch_after = 50

    config = get_residual_cov_experiment(exp)
    title = 'mincorig_residual_cov'

    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           # preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )

    run_finetune(ResNet50_o2, mincorig_finetune, input_shape, config, image_gen,
                 nb_classes=nb_classes,
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 title=title, verbose=(2, 1))


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
        model.name = 'baseline_resnet_with_1x1'
    else:
        return
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           )
    mincorig_finetune(model, image_gen=image_gen, title='mincorig_finetune_resnet',
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
                                 freeze_conv=True, cov_regularizer=cov_regularizer,
                                 last_conv_feature_maps=config.last_conv_feature_maps)

                mincorig_finetune(model, title='minc_orig_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                                  nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                                  image_gen=image_gen,
                                  batch_size=32, early_stop=early_stop,
                                  log=False,
                                  verbose=1)


def test_routine_resnet(config):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    nb_epoch_finetune = 2
    nb_epoch_after = 50
    title = 'test_minc_orig_cov'


    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           # preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )

    run_finetune(ResNet50_o2, mincorig_finetune, input_shape, config, image_gen,
                 nb_classes=nb_classes,
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 title=title, verbose=(2,1))


if __name__ == '__main__':
    # config = get_von_settings(4)
    # config = get_log_experiment(2)
    # config = get_experiment_settings(6)
    # config = get_aaai_experiment(1)

    # config = get_von_settings(4)
    # config = get_constraints_settings(1)
    # config = get_experiment_settings(7)
    # config = get_experiment_settings()
    # config = get_von_with_regroup(1)
    exp = 1
    config = get_VGG_testing_ideas(exp)

    # config = get_von_with_multibranch(2)
    # config.cov_mode = 'pmean'
    config.batch_size = 32
    # config = get_residual_cov_experiment(2)
    print(config.title)
    # log_model_to_path(ResNet50_o2, input_shape, config, nb_classes, 'minc2500')

    # run_routine_resnet(config, verbose=(1,1), nb_epoch_after=50, nb_epoch_finetune=2)

    # run_routine_resnet(config, verbose=(1, 2), stiefel_observed=[],
    # run_routine_resnet(config, verbose=(1, 2), stiefel_observed=['o2t'],
    #                    stiefel_lr=(0.01, 0.001), nb_epoch_finetune=2, nb_epoch_after=50)

    # run_routine_resnet_multibranch(config, verbose=(1, 2), stiefel_observed=['o2t'],
                                   # stiefel_lr=(0.01, 0.001), nb_epoch_finetune=2, nb_epoch_after=50)
    # run_routine_resnet_multibranch(config, verbose=(1, 2), nb_epoch_finetune=2, nb_epoch_after=50)

    run_routine_vgg(config, verbose=(2, 2),
                    # stiefel_observed=['o2t'], stiefel_lr=(0.01, 0.001),
                    nb_epoch_finetune=5, nb_epoch_after=50)

    # run_routine_resnet(config)

    # run_residual_cov_resnet(1)
    # baseline_finetune_resnet(2)
    # run_routine_vgg(4)
    # baseline_finetune_vgg()
    # test_routine_vgg(4)
    # test_routine_resnet(6)
    # test_routine_resnet(config)
