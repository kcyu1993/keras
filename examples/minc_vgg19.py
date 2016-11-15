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

from keras.layers import SecondaryStatistic, WeightedProbability
from keras.layers import Flatten, Dense, Dropout, O2Transform
from keras.layers import Input
from keras.models import Model, Sequential
from keras import regularizers
# from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.utils.model_utils import *
from keras.utils.data_utils import *
from keras.utils.logger import Logger
from keras.datasets.minc import Minc2500, MincOriginal
from example_engine import ExampleEngine

# Load the model
from keras.applications.vgg19 import VGG19, VGG19_bottom
from keras.applications.resnet50 import ResNet50MINC

import sys

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

### FOR model 1
INPUT_SHAPE=(3, 224, 224)


def create_ResNet50(second=False, parametric=True):
    if not second:
        model = ResNet50MINC(weights='imagenet', nb_class=NB_CLASS)
        model.name = 'ResNet50_original'
    else:
        x, weight_path, img_input = ResNet50MINC(include_top=False, nb_class=NB_CLASS, weights='imagenet')
        x = SecondaryStatistic(output_dim=None, parametrized=False, init='normal')(x)
        if parametric:
            x = O2Transform(output_dim=100, activation='relu')(x)
        x = WeightedProbability(output_dim=NB_CLASS, activation='softmax')(x)
        model = Model(input=img_input,
                      output=[x])
    return model

def create_VGG_snd():
    x, weight_path, img_input = VGG19_bottom(include_top=False, weights='imagenet')
    x = SecondaryStatistic(output_dim=None, parametrized=False, init='normal')(x)
    x = WeightedProbability(output_dim=NB_CLASS, activation='softmax')(x)
    return x, weight_path, img_input


def create_VGG_original2():
    x, weight_path, img_input = VGG19_bottom(include_top=False, weights='imagenet')
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(NB_CLASS, activation='softmax', name='predictions')(x)
    return x, weight_path, img_input


def create_alex_original():
    return VGG19(weights='imagenet')


def check_print(name='second'):
    # Create the Model
    if name is 'second':
        x, img_input = create_VGG_snd()
    elif name is 'original':
        x, img_input = create_VGG_original2()
    elif name is 'vgg':
        model = VGG19(weights='imagenet')

    # Create a Keras Model - Functional API
    # model = Model(input=img_input,
    #               output=[x])
    model.summary()
    # model.load_weights(os.path.join( , by_name=True))
    # Save a PNG of the Model Build
    # plot(model, to_file='./Model/AlexNet_{}.png'.format(name))

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


def test_model(model, title="", load_w=True, save_w=True, verbose=2):
    print("loading model from generator ")
    # tr, va, te = loads()
    loader = Minc2500()

    tr_iterator = loader.generator(input_file='train1.txt', batch_size=16, target_size=(INPUT_SHAPE[1], INPUT_SHAPE[2]))
    te_iterator = loader.generator(input_file='test1.txt', target_size=(INPUT_SHAPE[1], INPUT_SHAPE[2]))

    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    engine = ExampleEngine(data=tr_iterator, model=model, validation=te_iterator,
                           load_weight=load_w, save_weight=save_w, title=title)
    history = engine.fit()


def test_minc_original_VGG_reduced():
    x, img_input,  = create_VGG_original2()
    model = Model(input=img_input,
                  output=[x])
    test_model(model, title='minc_original_vgg', save_w=True, load_w=True)


def test_minc_original_VGG_generator(load=False, save=True, verbose=1):
    """
    This is testing case for minc dataset on original Alexnet as entry-point
        Experiment result:
            Not converging from random-initialization
            Extremely slow(or I dont know what is the relatively good speed)
            Batch size 128, training time is around 8s


    :return:
    """

    x, weight_path, img_input = VGG19_bottom(include_top=True, weights='imagenet', output_dim=NB_CLASS)
    model = Model(input=img_input,
                  output=[x])
    test_model(model, title='original_vgg', load_w=load, save_w=save, verbose=verbose)


def test_minc_snd_VGG_generator(load=False, save=True, verbose=1):
    x, weight_path, img_input = VGG19_bottom(weights='imagenet', output_dim=NB_CLASS)
    # After max pooling here
    x = SecondaryStatistic()(x)
    x = O2Transform(output_dim=1000)(x)
    x = WeightedProbability(output_dim=NB_CLASS, activation='softmax')

    model = Model(input=img_input,
                  output=[x])
    test_model(model, title='minc2500-vgg_snd', load_w=load, save_w=save, verbose=verbose)


def test_minc_ResNet50_generator(load=False, save=True, verbose=1):
    model = create_ResNet50()
    if load:
        weight_path = get_weight_path(model.name + "_minc2500.weights", dir='dataset')
    else:
        weight_path = None
    test_model(model, title='minc2500-res50_snd', load_w=load, save_w=save, verbose=verbose)


# def test_minc_VGG_generator(model, weight_path=None, load=False, save=True, verbose=1):
#     """
#     This is testing case for minc dataset on original Alexnet as entry-point
#         Experiment result:
#             Not converging from random-initialization
#             Extremely slow(or I dont know what is the relatively good speed)
#             Batch size 128, training time is around 8s
#
#
#     :return:
#     """
#     print("loading model from generator ")
#     # tr, va, te = loads()
#     loader = Minc2500()
#
#     tr_iterator = loader.generator(input_file='train1.txt', batch_size=16, target_size=(INPUT_SHAPE[1], INPUT_SHAPE[2]))
#     te_iterator = loader.generator(input_file='test1.txt', target_size=(INPUT_SHAPE[1], INPUT_SHAPE[2]))
#
#     tr_sample = tr_iterator.nb_sample
#     te_sample = te_iterator.nb_sample
#
#     if load:
#         weight_path = get_weight_path(model.name + "_minc2500.weights", dir='dataset')
#     if weight_path is not None:
#         model.load_weights(weight_path, by_name=True)
#     if verbose == 1:
#         model.summary()
#     model.compile(optimizer='rmsprop',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     model.fit_generator(tr_iterator, samples_per_epoch=128*200, nb_epoch=NB_EPOCH, nb_worker=4,
#                         validation_data=te_iterator, nb_val_samples=te_sample,
#                         verbose=verbose)
#
#     print("save the model")
#     if save:
#         weight_path = get_weight_path(model.name + "_minc2500.weights", dir='dataset')
#         model.save_weights(weight_path)



def test_routine1():
    print("test routine 1")
    sys.stdout = Logger(LOG_PATH + '/minc_resnet50_original1.log')
    # test_fitnet_layer(load=True, verbose=1)
    test_minc_ResNet50_generator(load=True, verbose=2)

def test_routine2():
    sys.stdout = Logger(LOG_PATH + '/minc_VGG19_original1.log')
    test_minc_original_VGG_generator(load=False, save=True, verbose=1)

def test_minc_original_loader():
    save_dir = os.path.join(get_dataset_dir(), 'debug')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    loader = MincOriginal()
    gen = loader.generator(save_dir=save_dir)
    a = gen.next()


if __name__ == '__main__':
    test_minc_original_loader()
    # test_minc_original_VGG_generator()
    # test_routine1()
    # test_routine2()
    # test_minc_original_alexnet_reduced()
    # test_VGG()
    # check_print('original')
    # check_print('second')


def test_VGG():
    # model = VGG19(weights='imagenet')
    x, weight_path, img_input = VGG19_bottom(include_top=False, weights='imagenet')
    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[x])

    model.summary()
    # model.load_weights(os.path.join( , by_name=True))
    # Save a PNG of the Model Build
    # plot(model, to_file='./Model/AlexNet_{}.png'.format(name))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print("loading weights")
    model.load_weights(weight_path, by_name=True)
    print('Model Compiled')