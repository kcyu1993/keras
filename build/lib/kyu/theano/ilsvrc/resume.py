"""
Resume from training

"""
import os

import tensorflow as tf
from kyu.theano.general.config import DCovConfig
from keras.layers import SecondaryStatistic, O2Transform

import keras.backend as K
from keras.preprocessing.image import ImageDataGeneratorAdvanced

from kyu.utils.imagenet_utils import preprocess_image_for_imagenet

from kyu.models.vgg import VGG16_o1, VGG16_o2
from kyu.models.resnet import ResNet50_o1, ResCovNet50, ResNet50_o2, ResNet50_o2_multibranch


# Some constants
from kyu.theano.general.finetune import resume_finetune, resume_finetune_with_Stiefel_layer
from kyu.theano.ilsvrc.train import imagenet_finetune

# from kyu.theano.minc.configs import get_von_with_regroup, get_matrix_bp, get_von_settings,get_von_with_multibranch, \
#     get_log_experiment, get_aaai_experiment
from kyu.theano.ilsvrc.config import *

nb_classes = 1000
if K.backend() == 'tensorflow':
    input_shape=(224,224,3)
    K.set_image_dim_ordering('tf')
else:
    input_shape=(3,224,224)
    K.set_image_dim_ordering('th')

TARGET_SIZE = (224,224)
RESCALE_SMALL = 256


# RESNET_BASELINE_WEIGHTS_PATH = '/home/kyu/.keras/models/ImageNet_finetune_resnet-baseline_resnet_1.weights'
# VGG_BASELINE_WEIGHTS_PATH = '/home/kyu/.keras/models/baseline/ImageNet_finetune_vgg16-baseline_vgg16_1.weights'

def resume_model_with_config(model, config, title='ImageNet',
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
        resume_finetune(model, imagenet_finetune,
                        weights_path=weights_path,
                        nb_classes=nb_classes,
                        input_shape=input_shape, config=config,
                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                        image_gen=image_gen, title=title, verbose=verbose,
                        monitor_classes=monitor_class,
                        monitor_measures=monitor_metrics,
                        by_name=by_name)
    else:
        resume_finetune_with_Stiefel_layer(model, imagenet_finetune,
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
                             title='ImageNet_resnet',
                             verbose=verbose, image_gen=image_gen,
                             nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                             stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed,
                             by_name=by_name)


def resume_routine_vgg(config, weights_path,
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
    resume_model_with_config(VGG16_o2, config,
                             weights_path=weights_path,
                             title='ImageNet_VGG',
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
                             title='ImageNet_resnet_MB',
                             verbose=verbose, image_gen=image_gen,
                             nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                             stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed,
                             by_name=by_name)


if __name__ == '__main__':
    # weights_path = 'finetune_minc2500_resnet_MB-stiefelminc_von_mean_o2t_no_wv_robost_None_cov_o2t_no_wv_wv64_pmean' \
    #                '-ResNet_o2-multi-o2t_no_wv_para-257_128_64_mode_3nb_branch_2_1.weights'
    # weights_path = 'retrain_minc2500_resnet50_stiefelaaai_baseline_cov_aaai_wv128_mean-' \
    #                'ResNet_o2_aaai_para-513_257_129_mode_1_1.weights'
    # weights_path = os.path.join('/home/kyu/.keras/models', weights_path)
    # weights_path = '/tmp/VGG16_o2_para-257_128_64_mode_1nb_branch_2_369_finetune.weights'
    # weights_path = '/home/kyu/.keras/models/finetune_minc2500_DR_pmean_robost_LC[]_cov_o2transform_wv64_pmean-' \
    #                'VGG16_o2_para-257_128_64_mode_1nb_branch_2_1.weights'
    # weights_path = VGG_BASELINE_WEIGHTS_PATH
    weights_path = 'finetune_ImageNet_VGG16_o2t_no_wv_robost_LC[1024]_exp_1_concat_cov_o2t_no_wv_wv64_pmean' \
                   '-VGG16_o2_para-257_128_64_mode_1nb_branch_4_1.weights.tmp'
    weights_path = os.path.join('/home/kyu/.keras/models/baseline', weights_path)
    # config = get_von_settings(4)
    # config = get_log_experiment(2)
    # config = get_experiment_settings(6)
    # weights_path = RESNET_BASELINE_WEIGHTS_PATH
    # config = get_aaai_experiment(2)

    # config = get_von_settings(4)
    # config = get_constraints_settings(1)
    # config = get_experiment_settings(7)
    # config = get_experiment_settings()
    # config = get_von_with_regroup(2)
    # config = get_VGG_dimension_reduction(1)
    # config = get_von_with_multibranch(3)

    config = get_VGG_testing_ideas(1)
    # config = get_aaai_experiment(1)
    # config.cov_mode = 'pmean'
    config.batch_size = 32 * 2
    # config = get_residual_cov_experiment(2)
    print(config.title)

    # resume_routine_resnet_multibranch(config, verbose=(1, 1), weights_path=weights_path, stiefel_observed=['o2t'],
    #                                   stiefel_lr=(0.001, 0.0001), nb_epoch_finetune=2, nb_epoch_after=100,
    #                                   by_name=True)
    # resume_routine_resnet(config=config, verbose=(1,1), weights_path=weights_path, by_name=True, nb_epoch_finetune=1,
    #                       stiefel_observed=['o2t'],
    #                       stiefel_lr=(0.001, 0.0001),
    #                       nb_epoch_after=100)
    resume_routine_vgg(config=config, verbose=(1,2), weights_path=weights_path, by_name=True, nb_epoch_finetune=1,
                       # stiefel_observed=['o2t'],
                       # stiefel_lr=(0.001, 0.0001),
                       nb_epoch_after=1000)

