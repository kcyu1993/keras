"""
Training with DTD dataset
"""
import os

from kyu.theano.general.config import DCovConfig

os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['KERAS_BACKEND'] = 'theano'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from kyu.utils.imagenet_utils import preprocess_image_for_imagenet

from kyu.legacy.vgg16 import VGG16_o1
from kyu.legacy.resnet50 import ResNet50_o1, ResCovNet50
from kyu.models.fitnet import fitnet_o1, fitnet_o2
from kyu.datasets.dtd import load_dtd
from kyu.theano.general.train import fit_model_v2

import keras.backend as K
from keras.preprocessing.image import ImageDataGeneratorAdvanced

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


def get_experiment_settings(exp=1):
    cov_regularizer = None
    nb_branch = 1
    dropout = False
    early_stop = True
    if exp == 1:
        params = [[], [64], [128], [100,50], [128, 64], [512, 256], [128,128,128]]
        mode_list = [1]
        cov_outputs = [128, 64, 47, 32]
        cov_branch = 'o2transform'
        cov_mode = 'channel'
        dropout = False
    elif exp == 2:
        params = [[], [64], [128], [100,50], [128, 64], [512, 256], [128,128,128]]
        mode_list = [1]
        cov_outputs = [128, 64, 47, 32]
        cov_branch = 'o2transform'
        cov_mode = 'mean'
        dropout = False
    elif exp == 3:
        params = [[], [64], [128], [100,50], [128, 64], [512, 256], [128,128,128]]
        mode_list = [1]
        cov_outputs = [128, 64, 47, 32]
        cov_branch = 'o2transform'
        cov_mode = 'mean'
        dropout = True
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
        params = [[64,64]]
        mode_list = [3]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True

    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        dropout=dropout,
                         nb_branch=nb_branch)
    return config


def dtd_fitting(model, image_gen=None, batch_size=32, title='dtd',
                optimizer=None, early_stop=False, verbose=2):
    """
    Fitting for DTD dataset
    Parameters
    ----------
    model

    Returns
    -------

    """
    train, _, test = load_dtd(True, image_gen=image_gen, batch_size=batch_size)
    fit_model_v2(model, [train, test], batch_size=batch_size, title=title,
                 nb_epoch=100,
                 optimizer=optimizer,
                 early_stop=early_stop,
                 verbose=verbose)


def baseline_VGG():
    model = VGG16_o1(denses=[1024], nb_classes=nb_classes, input_shape=input_shape)
    model.name = 'baseline_vgg16'
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                 horizontal_flip=True,
                                 preprocessing_function=preprocess_image_for_imagenet
                                 # channelwise_std_normalization=True
                                 )
    dtd_fitting(model, image_gen)


def baseline_fitnet():
    model = fitnet_o1(denses=[500], nb_classes=nb_classes, input_shape=input_shape, dropout=True, version=1)
    dtd_fitting(model, batch_size=32, title='dtd_fitnet_v1_o1_baseline')


def baseline_ResNet():
    model = ResNet50_o1(denses=[], nb_classes=nb_classes, input_shape=input_shape)
    model.name = 'baseline_resnet'
    dtd_fitting(model, batch_size=8, title='dtd_resnet50', early_stop=True)


def ResCovNet_test():
    model = ResCovNet50(parametrics=[], nb_classes=nb_classes, input_shape=input_shape, mode=1)
    model.name = 'ResCovNet non parametric'
    dtd_fitting(model, batch_size=8)
    dtd_fitting()


def run_routine1(exp=3):
    """
    Test Fitnet v1 o2 different settings

    Experiments :
        1. 2017.2.15    Test DCov 0,1,2,3 with different settings.
        2. 2017.2.15    Test DCov with combined matrix.

    Returns
    -------

    """
    nb_epoch = 200
    # exp = 3

    # if exp == 1:
    #     params = [[], [64], [128], [100,50], [128, 64], [512, 256], [128,128,128]]
    #     mode_list = [1]
    #     cov_outputs = [128, 64, 47, 32]
    #     cov_branch = 'o2transform'
    #     cov_mode = 'channel'
    #     dropout = False
    # elif exp == 2:
    #     params = [[], [64], [128], [100,50], [128, 64], [512, 256], [128,128,128]]
    #     mode_list = [1]
    #     cov_outputs = [128, 64, 47, 32]
    #     cov_branch = 'o2transform'
    #     cov_mode = 'mean'
    #     dropout = False
    # elif exp == 3:
    #     params = [[], [64], [128], [100,50], [128, 64], [512, 256], [128,128,128]]
    #     mode_list = [1]
    #     cov_outputs = [128, 64, 47, 32]
    #     cov_branch = 'o2transform'
    #     cov_mode = 'mean'
    #     dropout = True
    # else:
    #     return

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
                model = fitnet_o2(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                  version=1,
                                  nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape,
                                  dropout=config.dropout)

                dtd_fitting(model, title='dtd_cov_{}_wv{}_{}_mode{}'.format(cov_branch,
                                                                            str(cov_output), cov_mode, mode),
                            batch_size=32,
                            early_stop=early_stop)


def run_routine1_resnet():
    nb_epoch = 200
    exp = 2

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
    else:
        return

    print("Running experiment {}".format(exp))
    for param in params:
        for mode in mode_list:
            for cov_output in cov_outputs:
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                model = ResCovNet50(parametrics=param, mode=mode, cov_branch=cov_branch, cov_mode=cov_mode,
                                    nb_classes=nb_classes, cov_branch_output=cov_output, input_shape=input_shape)
                dtd_fitting(model, title='dtd_cov_{}_wv{}_{}_mode_{}'.format(
                    cov_branch, str(cov_output), cov_mode, mode),
                            batch_size=4, early_stop=early_stop)


if __name__ == '__main__':
    # baseline_ResNet()
    # ResCovNet_test()
    # baseline_VGG()
    # baseline_fitnet()
    run_routine1(5)
    # run_routine1_resnet()