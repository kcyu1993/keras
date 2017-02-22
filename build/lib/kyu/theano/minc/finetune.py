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
                 verbose=2):

    loader = Minc2500()
    train, test = load_minc2500(index=1, target_size=TARGET_SIZE, gen=image_gen, batch_size=batch_size)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    fit_model_v2(model, [train, test], batch_size=batch_size, title=title,
                 nb_epoch=nb_epoch_finetune,
                 optimizer=optimizer,
                 early_stop=early_stop,
                 verbose=verbose)
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
                     verbose=verbose)

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
        last_config_feature_maps = [256]
        batch_size = 32

    elif exp == 5:
        """ Test ResNet 50 """
        # params = [[256, 128]]
        params = [[1024, 512]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        # last_config_feature_maps = [1024, 512]
        batch_size = 4
    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size)
    return config


def run_routine_resnet(exp=1):
    """
    Finetune the ResNet-DCov

    Returns
    -------
m
    """
    nb_epoch_finetune = 15
    nb_epoch_after = 50

    config = get_experiment_settings(exp)
    cov_branch = config.cov_branch
    cov_mode = config.cov_mode
    cov_regularizer = config.cov_regularizer
    early_stop = config.early_stop

    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           # preprocessing_function=preprocess_image_for_imagenet
                                           # channelwise_std_normalization=True
                                           )

    print("Running experiment {}".format(exp))
    for param in config.params:
        for mode in config.mode_list:
            for cov_output in config.cov_outputs:
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                sess = K.get_session()
                with sess.as_default():
                    model = ResNet50_o2(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                        nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape,
                                        last_avg=False,
                                        freeze_conv=True,
                                        cov_regularizer=cov_regularizer,
                                        last_conv_feature_maps=config.last_conv_feature_maps)
                    minc2500_finetune(model, title='minc2500_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                                      nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune,
                                      batch_size=config.batch_size, early_stop=early_stop, verbose=2, image_gen=image_gen)
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
                    minc2500_finetune(model, title='minc2500_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                                      nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after,
                                      batch_size=4, early_stop=early_stop, verbose=2)

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


def baseline_finetune_resnet():
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

                minc2500_finetune(model, title='minc2500_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                                  nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                                  image_gen=image_gen,
                                  batch_size=32, early_stop=early_stop,
                                  verbose=1)


def test_routine_resnet(exp=1):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    nb_epoch_finetune = 1
    nb_epoch_after = 50

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
                sess = K.get_session()
                with sess.as_default():
                    model = ResNet50_o2(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                        nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape,
                                        last_avg=False,
                                        freeze_conv=True,
                                        cov_regularizer=cov_regularizer)
                    minc2500_finetune(model, title='minc2500_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                                 nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune,
                                 batch_size=4, early_stop=early_stop, verbose=1)
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
                    minc2500_finetune(model, title='minc2500_cov_{}_wv{}_{}'.format(cov_branch, str(cov_output), cov_mode),
                                      nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after,
                                      batch_size=4, early_stop=early_stop, verbose=1)


if __name__ == '__main__':
    run_routine_resnet(5)
    # baseline_finetune_res()
    # run_routine_vgg(4)
    # baseline_finetune_vgg()
    # test_routine_vgg(4)
    # test_routine_resnet(1)
