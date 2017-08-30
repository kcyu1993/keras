"""
Resume from training

"""
import os

from keras.layers import SecondaryStatistic, O2Transform

import keras.backend as K
from keras.preprocessing.image import ImageDataGeneratorAdvanced
from kyu.models.fitnet import fitnet_o2

from kyu.legacy.resnet50 import ResNet50_o1, ResNet50_o2_multibranch, ResNet50_o2, ResCovNet50

# Some constants
from kyu.theano.general.finetune import resume_finetune, resume_finetune_with_Stiefel_layer
from kyu.theano.cifar.train import cifar_train

from kyu.theano.cifar.configs import get_experiment_settings, get_aaai_experiment, get_log_experiment, get_von_settings, \
    get_von_with_regroup, get_von_with_multibranch, get_matrix_bp

nb_classes = 10
if K.backend() == 'tensorflow':
    input_shape=(32, 32, 3)
    K.set_image_dim_ordering('tf')
else:
    input_shape=(3,32, 32)
    K.set_image_dim_ordering('th')

TARGET_SIZE = (32, 32)
RESCALE_SMALL = 32


# CIFAR_BASELINE_WEIGHTS_PATH = '/home/kyu/.keras/models/cifar10-baseline-fitnet_v2_para-16_32_mode_0_0.weights'
CIFAR_BASELINE_WEIGHTS_PATH = '/home/kyu/.keras/models/baseline/cifar10_fitnet_v1_baseline-fitnet_v1_dense-500.weights'


def resume_model_with_config(model, config, title='cifar10',
                             image_gen=None, weights_path='',
                             verbose=(2, 2), nb_epoch_finetune=0, nb_epoch_after=50,
                             stiefel_observed=None, stiefel_lr=0.0001,
                             by_name=False):
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
    monitor_metrics = ['matrix_image', ]
    if stiefel_observed is None:
        resume_finetune(model, cifar_train,
                        weights_path=weights_path,
                        nb_classes=nb_classes,
                        input_shape=input_shape, config=config,
                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                        image_gen=image_gen, title=title, verbose=verbose,
                        monitor_classes=monitor_class,
                        monitor_measures=monitor_metrics,
                        by_name=by_name)
    else:
        resume_finetune_with_Stiefel_layer(model, cifar_train,
                                           weights_path=weights_path,
                                           nb_classes=nb_classes,
                                           input_shape=input_shape, config=config,
                                           nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                                           image_gen=image_gen, title=title+'-stiefel', verbose=verbose,
                                           monitor_classes=monitor_class,
                                           monitor_measures=monitor_metrics,
                                           observed_keywords=stiefel_observed,
                                           lr=stiefel_lr,
                                           by_name=by_name)


def resume_routine_resnet(config, weights_path,
                          verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                          stiefel_observed=None, stiefel_lr=0.01,
                          by_name=False):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           )
    resume_model_with_config(ResNet50_o2, config,
                             weights_path=weights_path,
                             title='cifar10_resnet',
                             verbose=verbose, image_gen=image_gen,
                             nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                             stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed,
                             by_name=by_name)


def resume_routine_resnet_multibranch(config, verbose=(2,2),
                                      weights_path='',
                                      nb_epoch_finetune=0, nb_epoch_after=50,
                                      stiefel_observed=None, stiefel_lr=0.01,
                                      by_name=False):
    """
        Finetune the ResNet-DCov

        Returns
        -------

        """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           )
    resume_model_with_config(ResNet50_o2_multibranch, config,
                             weights_path=weights_path,
                             title='cifar10_resnet_MB',
                             verbose=verbose, image_gen=image_gen,
                             nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                             stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed,
                             by_name=by_name)


def resume_routine_fitnet(config, verbose=(2,2),
                          weights_path='',
                          nb_epoch_finetune=0, nb_epoch_after=50,
                          stiefel_observed=None, stiefel_lr=0.01,
                          by_name=False):
    # image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
    #                                        horizontal_flip=True,
    #                                        )
    resume_model_with_config(fitnet_o2, config,
                             weights_path=weights_path,
                             title=config.title,
                             verbose=verbose, image_gen=None,
                             nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                             stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed,
                             by_name=by_name)


if __name__ == '__main__':
    # weights_path = 'finetune_cifar10_resnet_MB-stiefelminc_von_mean_o2t_no_wv_robost_None_cov_o2t_no_wv_wv64_pmean' \
    #                '-ResNet_o2-multi-o2t_no_wv_para-257_128_64_mode_3nb_branch_2_1.weights'
    # weights_path = os.path.join('/home/kyu/.keras/models', weights_path)
    # config = get_von_settings(4)
    # config = get_log_experiment(2)
    # config = get_experiment_settings(6)
    weights_path = CIFAR_BASELINE_WEIGHTS_PATH
    # config = get_aaai_experiment(1)
    config = get_matrix_bp(1)
    # config = get_experiment_settings(4)

    # config = get_von_settings(4)
    # config = get_constraints_settings(1)
    # config = get_experiment_settings(7)
    # config = get_experiment_settings()
    # config = get_von_with_regroup(2)
    # config = get_von_with_multibranch(3)
    config.batch_size = 128
    # config = get_residual_cov_experiment(2)
    print(config.title)

    resume_routine_fitnet(config=config, verbose=(2,2), weights_path=weights_path, by_name=True, nb_epoch_finetune=30,
                          nb_epoch_after=100)

    # resume_routine_resnet_multibranch(config, verbose=(1, 1), weights_path=weights_path, stiefel_observed=['o2t'],
    #                                   stiefel_lr=(0.001, 0.0001), nb_epoch_finetune=2, nb_epoch_after=100,
    #                                   by_name=True)
    # resume_routine_resnet(config=config, verbose=(1,1), weights_path=weights_path, by_name=True, nb_epoch_finetune=100,
    #                       nb_epoch_after=1)