"""
Finetune with MINC dataset
"""
import os
import warnings

import keras.backend as K
from keras.engine import merge
from keras.layers import Flatten

from keras.applications import ResNet50
from keras.layers import Dense
from kyu.models.keras_support import covariance_block_original, covariance_block_vector_space
from kyu.theano.general.config import DCovConfig
from kyu.theano.general.finetune import run_finetune

os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from kyu.datasets.imagenet import preprocess_image_for_imagenet

from kyu.models.vgg import VGG16_o1, VGG16_o2
from kyu.models.resnet import ResNet50_o1, ResCovNet50, ResNet50_o2
from kyu.models.fitnet import fitnet_v1_o1, fitnet_v1_o2

from kyu.datasets.minc import Minc2500, load_minc2500

from kyu.theano.general.train import fit_model_v2, toggle_trainable_layers, Model

from kyu.models.minc import minc_fitnet_v2

import keras.backend as K
from keras.preprocessing.image import ImageDataGeneratorAdvanced, ImageDataGenerator

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


def minc2500_finetune(model,
                      nb_epoch_finetune=100, nb_epoch_after=0, batch_size=32,
                      image_gen=None,
                      title='MINC2500_finetune', early_stop=False,
                      keyword='',
                      optimizer=None,
                      log=True,
                      verbose=2,
                      lr=0.001):

    loader = Minc2500()
    train, test = load_minc2500(index=1, target_size=TARGET_SIZE, gen=image_gen, batch_size=batch_size)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    fit_model_v2(model, [train, test], batch_size=batch_size, title=title,
                 nb_epoch=nb_epoch_finetune,
                 optimizer=optimizer,
                 early_stop=early_stop,
                 verbose=verbose,
                 log=log,
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
                     lr=lr/10)

    return


def get_experiment_settings(exp=1):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
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
        params = [[256, 128]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
        # last_config_feature_maps = [256]
        batch_size = 32

    elif exp == 5:
        """ Test ResNet 50 """
        params = [[256, 128], [256, 128, 64], [256], [128], [256, 256, 128],]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
        batch_size = 32
    elif exp == 6:
        """ Test Multi branch ResNet 50 """
        # nb_branch = 4
        nb_branch = 8
        params = [[256, 128], [256, 128, 64],]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128, 64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        # last_config_feature_maps = [1024, 512]
        batch_size = 32
    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp)
    return config


def get_aaai_experiment(exp):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    if exp == 1:
        """ aaai paper test """
        nb_branch = 1
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [[513, 257, 129], [513, 513, 513]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'aaai'
        early_stop = False
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
        batch_size = 32
        vectorization = 'dense'
    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, vectorization=vectorization, epsilon=1e-5)
    return config


def get_log_experiment(exp):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    if exp == 1:
        """ log test """
        nb_branch = 1
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [[513, 257, 129], [513, 513, 513]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'log'
        early_stop = False
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
        batch_size = 32
        vectorization = 'wv'
    elif exp == 2:
        """ aaai paper test """
        nb_branch = 1
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [[256, 128, 64], [513, 257, 129, 64]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'log'
        early_stop = False
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
        batch_size = 32
        vectorization = 'dense'
    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, vectorization=vectorization, epsilon=1e-5)
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
        early_stop = False
        # cov_regularizer = None
        # cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
        robust = True
        cov_alpha = 0.75
    elif exp == 2:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 4
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = False

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
        cov_regularizer = None
        # cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
        robust = True
        cov_alpha = 0.75
    elif exp == 4:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 4
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        cov_regularizer = None
        # cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, robust=robust, cov_alpha=cov_alpha)
    return config


def get_residual_cov_experiment(exp):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    if exp == 1:
        """ Test Multi branch ResNet 50 """
        nb_branch = 1
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [ [513, 513,], [513, 513,513, 513],[513, 513, 513, 513, 513,],[513, 513, 513, 513, 513, 513], ]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'residual'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
        batch_size = 32
    elif exp == 2:
        """ Test Multi branch ResNet 50 """
        nb_branch = 4
        params = [[513, 513, 513], [256, 256, 256]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'residual'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        # last_config_feature_maps = [1024]
        batch_size = 32
    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp)
    return config


def run_routine_resnet(config, nb_epoch_finetune=50, nb_epoch_after=50):
    """
    Finetune the ResNet-DCov

    Returns
    -------
m
    """
    title = 'minc2500_cov'

    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           )

    run_finetune(ResNet50_o2, minc2500_finetune, input_shape, config, image_gen,
                 nb_classes=nb_classes,
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 title=title, verbose=(2, 2))


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

                minc2500_finetune(model, title='minc2500_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
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
    minc2500_finetune(model, image_gen=image_gen, title='minc2500_finetune_vgg16',
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
    title = 'minc2500_residual_cov'

    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           # preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )

    run_finetune(ResNet50_o2, minc2500_finetune, input_shape, config, image_gen,
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
    minc2500_finetune(model, image_gen=image_gen, title='minc2500_finetune_resnet',
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

                minc2500_finetune(model, title='minc2500_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
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
    title = 'test_minc2500_cov'


    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           # preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )

    run_finetune(ResNet50_o2, minc2500_finetune, input_shape, config, image_gen,
                 nb_classes=nb_classes,
                 nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                 title=title, verbose=(2,1))


if __name__ == '__main__':
    config = get_von_settings(4)
    # config = get_log_experiment(2)
    # config = get_experiment_settings(6)
    # config = get_aaai_experiment(1)
    # run_routine_resnet(config)
    # run_residual_cov_resnet(1)
    # baseline_finetune_resnet(2)
    # run_routine_vgg(4)
    # baseline_finetune_vgg()
    # test_routine_vgg(4)
    # test_routine_resnet(6)
    # test_routine_resnet(config)
    run_routine_resnet(config, 10, 50)


