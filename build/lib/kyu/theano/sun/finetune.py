"""
Finetune with MINC dataset
"""
import os

from kyu.models.secondstat import SecondaryStatistic, O2Transform
from kyu.datasets.sun import SUN397
from kyu.theano.general.config import DCovConfig
from kyu.theano.general.finetune import run_finetune, run_finetune_with_Stiefel_layer, finetune_model_with_config
from kyu.theano.sun.configs import get_experiment_settings, get_von_settings, get_residual_cov_experiment, \
    get_VGG_dimension_reduction, get_matrix_bp, get_VGG_testing_ideas, get_cov_alpha_cv, get_cov_beta_cv, \
    get_aaai_experiment, get_ResNet_testing_ideas, get_iccv_experiment
from kyu.theano.mit.configs import get_new_experiment

os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from kyu.utils.imagenet_utils import preprocess_image_for_imagenet

from kyu.models.vgg import VGG16_bilinear
from kyu.legacy.vgg16 import VGG16_o1, VGG16_o2
from kyu.legacy.resnet50 import ResNet50_o1, ResNet50_o2_multibranch, ResNet50_o2

from kyu.theano.general.train import fit_model_v2, Model
from kyu.utils.train_utils import toggle_trainable_layers
from kyu.utils.image import ImageDataGeneratorAdvanced

import keras.backend as K

# Some constants
nb_classes = 397
if K.backend() == 'tensorflow':
    input_shape = (224,224,3)
    K.set_image_dim_ordering('tf')
else:
    input_shape = (3,224,224)
    K.set_image_dim_ordering('th')

TARGET_SIZE = (224,224)
RESCALE_SMALL = 270


RESNET_BASELINE_WEIGHTS_PATH = '/home/kyu/.keras/models/sun_resnet50-baseline_resnet_1.weights'


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


def sun_finetune(model,
                 nb_epoch_finetune=100, nb_epoch_after=0, batch_size=32,
                 image_gen=None,
                 title='sun_finetune', early_stop=False,
                 keyword='',
                 optimizer=None,
                 log=True,
                 lr_decay=True,
                 weight_path='',
                 load=False,
                 verbose=2,
                 lr=0.001):
    lr_decay = False
    loader = SUN397(dirpath='/home/kyu/.keras/datasets/sun')
    # train = loader.generator(mode='train', target_size=TARGET_SIZE, image_data_generator=image_gen, batch_size=batch_size)
    train = loader.generator(mode='train', target_size=TARGET_SIZE,
                             image_data_generator=image_gen, batch_size=batch_size)
    test = loader.generator(mode='test', target_size=TARGET_SIZE,
                            image_data_generator=image_gen, batch_size=batch_size)
    # if optimizer is None:
    # model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    fit_model_v2(model, [train, test], batch_size=batch_size*4, title=title,
                 nb_epoch=nb_epoch_finetune,
                 optimizer=optimizer,
                 early_stop=early_stop,
                 verbose=verbose,
                 lr_decay=lr_decay,
                 weight_path=weight_path,
                 load=load,
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
                     lr_decay=lr_decay,
                     lr=lr/10)

    return


def run_model_with_config(model, config, title='sun',
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
    finetune_model_with_config(model, sun_finetune, config, nb_classes, input_shape,
                               title=title, image_gen=image_gen, verbose=verbose,
                               nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                               stiefel_observed=stiefel_observed, stiefel_lr=stiefel_lr)


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
    run_model_with_config(ResNet50_o2, config, title='sun_resnet',
                          verbose=verbose, image_gen=image_gen,
                          nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                          stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed)
    # if stiefel_observed is None:
    #     run_finetune(ResNet50_o2, sun_finetune,
    #                  nb_classes=nb_classes,
    #                  input_shape=input_shape, config=config,
    #                  nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
    #                  image_gen=image_gen, title='sun_resnet50', verbose=verbose,
    #                  monitor_classes=monitor_class,
    #                  monitor_measures=monitor_metrics)
    # else:
    #     run_finetune_with_Stiefel_layer(ResNet50_o2, sun_finetune,
    #                                     nb_classes=nb_classes,
    #                                     input_shape=input_shape, config=config,
    #                                     nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
    #                                     image_gen=image_gen, title='sun_resnet50_stiefel', verbose=verbose,
    #                                     monitor_classes=monitor_class,
    #                                     monitor_measures=monitor_metrics,
    #                                     observed_keywords=stiefel_observed,
    #                                     lr=stiefel_lr)


def run_routine_vgg(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                    stiefel_observed=None, stiefel_lr=0.01):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           width_shift_range=0.2,
                                           zoom_range=[0.7,1.3],
                                           height_shift_range=0.2,
                                           # preprocessing_function=preprocess_image_for_imagenet
                                           )

    monitor_class = (O2Transform, SecondaryStatistic)
    # monitor_metrics = ['weight_norm',]
    # monitor_metrics = ['output_norm',]
    monitor_metrics = ['matrix_image',]
    run_model_with_config(VGG16_o2, config, title='sun_VGG16',
                          verbose=verbose, image_gen=image_gen,
                          nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                          stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed)


def run_routine_resnet_multibranch(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                                   stiefel_observed=None, stiefel_lr=0.01):
    """
        Finetune the ResNet-DCov

        Returns
        -------

        """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           )

    run_model_with_config(ResNet50_o2_multibranch, config, title='sun_resnet_MB',
                          verbose=verbose, image_gen=image_gen,
                          nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                          stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed)


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
    title = 'sun_residual_cov'

    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           # preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )

    run_finetune(ResNet50_o2, sun_finetune, input_shape, config, image_gen,
                 nb_classes=nb_classes,
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 title=title, verbose=(2, 1))


def baseline_finetune_resnet(exp):
    nb_epoch_finetune = 5
    nb_epoch_after = 100
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
    sun_finetune(model, image_gen=image_gen, title='sun_finetune_resnet',
                      nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                      batch_size=32, early_stop=True)


def baseline_finetune_vgg(exp=1):
    nb_epoch_finetune = 10
    nb_epoch_after = 50
    if exp == 1:
        model = VGG16_o1(denses=[1024, 1024], nb_classes=nb_classes, input_shape=input_shape, freeze_conv=True,
                         last_conv=False, load_weights=True, last_pooling=True,
                         )
        model.name = 'baseline_vgg16'
    elif exp == 2:
        model = VGG16_o1(denses=[1024, 1024], nb_classes=nb_classes, input_shape=input_shape, freeze_conv=True,
                         last_pooling=True,
                         last_conv=True)
        model.name = 'baseline_vgg16_with_1x1'
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL,
                                           random_crop=True,
                                           horizontal_flip=True,
                                           # width_shift_range=0.1,
                                           # height_shift_range=0.1,
                                           # preprocessing_function=preprocess_image_for_imagenet
                                           )
    sun_finetune(model, image_gen=image_gen, title='sun_finetune_vgg16',
                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                        batch_size=32, early_stop=False)


def baseline_finetune_bilinear(exp=1):
    nb_epoch_finetune = 1
    nb_epoch_after = 100

    if exp == 1:
        model = VGG16_bilinear(nb_class=nb_classes, input_shape=input_shape, freeze_conv=True)

    model.name = 'baseline_vgg16_bilinear'
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )
    weight_path = '/home/kyu/.keras/models/mit_finetune_vgg16_bilinear-baseline_vgg16_bilinear_1.weights_100'
    sun_finetune(model, image_gen=image_gen, title='mit_finetune_vgg16_bilinear',
                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                        weight_path=weight_path, load=True,
                        batch_size=32, early_stop=True, lr_decay=False,
                        lr=0.05)


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

                sun_finetune(model, title='sun_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
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
    title = 'test_sun_cov'

    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           # preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )

    run_finetune(ResNet50_o2, sun_finetune, input_shape, config, image_gen,
                 nb_classes=nb_classes,
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 title=title, verbose=(2,1))

def test_loader():
    loader = SUN397(dirpath='/home/kyu/cvkyu/dataset/sun')
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           preprocessing_function=preprocess_image_for_imagenet
                                           )
    train = loader.generator('train', batch_size=16,
                             image_data_generator=image_gen,
                             save_to_dir='/home/kyu/cvkyu/plots',
                             save_prefix='sun', save_format='png'
                             )
    train.next()


if __name__ == '__main__':
    # test_loader()
    exp = 5
    # config = get_von_settings(4)
    # config = get_log_experiment(2)
    # config = get_experiment_settings(6)
    # config = get_aaai_experiment(2)
    # config = get_VGG_dimension_reduction(4)

    # config = get_VGG_testing_ideas(exp)
    config = get_iccv_experiment(1)
    # config = get_ResNet_testing_ideas(exp)
    # config = get_von_settings(4)
    # config = get_constraints_settings(1)
    # config = get_experiment_settings(7)
    # config = get_experiment_settings()
    # config = get_von_with_regroup(1)
    # config = get_von_with_multibranch(1)

    # config = get_cov_alpha_cv(1)
    # config = get_cov_beta_cv(1)
    # config.batch_size = 32
    # baseline_finetune_bilinear(1)
    # config = get_new_experiment(6)
    # config = get_aaai_experiment(1)
    # config = get_matrix_bp(1)

    print(config.title)

    # config.cov_mode = 'channel'
    # config.batch_size = 16
    # config = get_residual_cov_experiment(2)
    # log_model_to_path(ResNet50_o2, input_shape, config, nb_classes, 'sun')
    # run_routine_resnet(config, verbose=(1, 2), nb_epoch_after=70, nb_epoch_finetune=4)
    # run_routine_resnet(config, verbose=(1, 2), stiefel_observed=[],
    # run_routine_resnet(config, verbose=(1, 1), stiefel_observed=['o2t'],
    #                    stiefel_lr=(0.001, 0.001), nb_epoch_finetune=2, nb_epoch_after=100)
    # run_routine_resnet_multibranch(config, verbose=(1, 1), stiefel_observed=['o2t'],
    #                                stiefel_lr=(0.01, 0.001), nb_epoch_finetune=5, nb_epoch_after=50)
    # run_routine_resnet_multibranch(config, verbose=(1, 2), stiefel_observed=None,
    #                                stiefel_lr=None, nb_epoch_finetune=4, nb_epoch_after=50)

    # config.title = 'minc_VGG_TEST_original_exp{}'.format(exp)
    run_routine_vgg(config, verbose=(2, 2),
                    # stiefel_observed=['o2t'], stiefel_lr=(0.001, 0.001),
                    nb_epoch_finetune=5, nb_epoch_after=200)
    #
    # run_routine_resnet(config)

    # run_residual_cov_resnet(1)
    # baseline_finetune_vgg(1)

    # baseline_finetune_resnet(2)
    # run_routine_vgg(4)
    # test_routine_vgg(4)
    # test_routine_resnet(6)
    # test_routine_resnet(config)

    """
    run with zoom rqnge 0.5 to 1.5
    """
