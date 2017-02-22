"""
    Model Name:

        AlexNet - using the Functional Keras API

        Replicated from the Caffe Zoo Model Version.



    Paper:

         ImageNet classification with deep convolutional neural networks by Krizhevsky et al. in NIPS 2012

    Alternative Example:

        Available at: http://caffe.berkeleyvision.org/model_zoo.html

        https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet

    Original Dataset:

        ILSVRC 2012

"""

import sys
import keras.backend as K

from keras_old.applications.vgg19 import VGG19_bottom

from kyu.utils.example_engine import ExampleEngine
from kyu.datasets.minc import Minc2500, MincOriginal

from keras.applications.resnet50 import ResNet50MINC, ResCovNet50MINC, ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import O2Transform
from keras.layers import SecondaryStatistic, WeightedVectorization
from keras.models import Model
from keras.utils.data_utils import *
from keras.utils.logger import Logger
from keras.utils.np_utils import to_categorical

LOG_PATH = get_absolute_dir_project('model_saved/log')
# global constants
NB_CLASS = 23         # number of classes
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 128
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005
NB_EPOCH = 20
LRN2D_norm = True       # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'th'

TARGET_SIZE=(224,224)
### FOR model 1
if K.backend() == 'tensorflow':
    INPUT_SHAPE = TARGET_SIZE + (3,)
    K.set_image_dim_ordering('tf')
else:
    INPUT_SHAPE=(3,) + TARGET_SIZE
    K.set_image_dim_ordering('th')


def check_print(model):
    # Create the Model
    model.summary()
    # model.load_weights(os.path.join( , by_name=True))
    # Save a PNG of the Model Build
    # plot(model, to_file='./Model/AlexNet_{}.png'.format(name))
    #
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print('Model Compiled')


def load():
    loader = Minc2500()
    # (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = loader.loadfromfile('minc-2500.plk.hdf5')
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = loader.loadfromfile('minc-2500.plk.hdf5')
    tr_Y = to_categorical(tr_y, nb_classes=NB_CLASS)
    va_Y = to_categorical(va_y, nb_classes=NB_CLASS)
    te_Y = to_categorical(te_y, nb_classes=NB_CLASS)
    return (tr_x, tr_Y), (va_x, va_Y), (te_x, te_Y)


def run_minc_original_model(model, title='', load_w=True, save_w=True, plot=False, verbose=2):
    """
    Run Test on MINC original dataset.

    Model should take a input as 224,224

    Parameters
    ----------
    model
    title
    load_w
    save_w
    plot
    verbose

    Returns
    -------

    """
    print("loading model from generator ")
    # tr, va, te = loads()
    loader = MincOriginal()

    tr_iterator = loader.generator(input_file='train.txt', batch_size=16, target_size=TARGET_SIZE)
    te_iterator = loader.generator(input_file='test.txt', target_size=TARGET_SIZE)

    # model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("initialize engine")
    engine = ExampleEngine(data=tr_iterator, model=model, validation=te_iterator,
                           load_weight=load_w, save_weight=save_w,
                           save_per_epoch=True, lr_decay=True,
                           title=title,
                           verbose=verbose)
    history = engine.fit(nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE)
    if plot:
        engine.plot_result('acc')
        engine.plot_result('loss')


def run_minc2500_model(model, title="", load_w=True, save_w=True, plot=False, verbose=2, imagegenerator=None):
    print("loading model from generator ")
    # tr, va, te = loads()
    loader = Minc2500()

    tr_iterator = loader.generator(input_file='train1.txt', batch_size=16, target_size=TARGET_SIZE, gen=imagegenerator)
    te_iterator = loader.generator(input_file='test1.txt', target_size=TARGET_SIZE, gen=imagegenerator)

    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    engine = ExampleEngine(data=tr_iterator, model=model, validation=te_iterator,
                           load_weight=load_w, save_weight=save_w, title=title)
    history = engine.fit(nb_epoch=NB_EPOCH)
    if plot:
        engine.plot_result()


def run_routine1():
    print("test routine 1")
    sys.stdout = Logger(LOG_PATH + '/minc_resnet50_original1.log')
    # test_fitnet_layer(load=True, verbose=1)
    model = ResNet50MINC(input_shape=INPUT_SHAPE)
    run_minc2500_model(model, 'minc_2500_ResNet50', verbose=1)


def run_routine2():
    sys.stdout = Logger(LOG_PATH + '/minc_VGG19_original1.log')
    # run_minc_original_VGG_generator(load=False, save=True, verbose=1)


def run_routine3(): # ResCov 1
    print('Non para ResCovNet 50')
    # run_minc_ResNet50_generator(load=True, save=True,
    #                             second=True, parametric=False,
    #                             verbose=2, plot=True)


def run_routine4():
    print("ResCovNet 50 cov_mode 0")
    # run_minc_original_rescovnet(load=True, save=True)


def run_routine5():
    from kyu.models.minc import minc_fitnet_v2
    model = minc_fitnet_v2()
    run_minc2500_model(model, title='minc-fitnet_v3', load_w=False)


def run_routine5_tf():
    from kyu.models.minc import minc_fitnet_v2
    import tensorflow as tf
    with tf.device('/gpu:0'):
        model = minc_fitnet_v2()
        run_minc2500_model(model, title='minc-fitnet_v2', load_w=False)

if __name__ == '__main__':
    # test_minc_original_loader()
    # run_minc_original_VGG_generator()
    NB_EPOCH = 150
    run_routine1()

    # run_routine3()
    # run_routine4()
    # run_routine5()
    # run_routine2()
    # test_minc_original_alexnet_reduced()
    # test_VGG()
    # check_print('original')
    # check_print('second')


0
