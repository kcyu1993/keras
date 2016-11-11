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

from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, SecondaryStatistic, WeightedProbability
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model, Sequential
from keras import regularizers
# from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.utils.model_utils import *
from keras.datasets.minc import Minc2500


# Load the model
from keras.applications.vgg19 import VGG19, VGG19_bottom

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
NB_EPOCH = 50
LRN2D_norm = True       # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'th'

### FOR model 1
INPUT_SHAPE=(3, 224, 224)


def create_VGG_snd():
    x, weight_path, img_input = VGG19_bottom(include_top=False, weights='imagenet')
    x = SecondaryStatistic(output_dim=None, parametrized=False, init='normal')(x)
    x = WeightedProbability(output_dim=NB_CLASS, activation='softmax')(x)
    return x, weight_path, img_input


def create_VGG_original2():
    x, weight_path, img_input = VGG19_bottom(include_top=False, weights='imagenet')
    x = Flatten()(x)
    x = Dense(output_dim=NB_CLASS,
              activation='softmax')(x)
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


def test_minc_original_VGG_reduced():
    print("loads reduced model from raw image data")
    loader = Minc2500()

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = loader.loadwithsplit()
    tr_Y = to_categorical(tr_y, nb_classes=NB_CLASS)
    va_Y = to_categorical(va_y, nb_classes=NB_CLASS)
    te_Y = to_categorical(te_y, nb_classes=NB_CLASS)

    x, img_input,  = create_VGG_original2()
    model = Model(input=img_input,
                  output=[x])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(tr_x, tr_Y,
              batch_size=32,
              nb_epoch=NB_EPOCH,
              shuffle='batch',
              validation_data=(va_x, va_Y),
              verbose=1)
    score = model.evaluate(te_x, te_Y, verbose=0)
    print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))



def test_minc_VGG_snd_generator():
    """
    This is testing case for minc dataset on original Alexnet as entry-point
        Experiment result:
            Not converging from random-initialization
            Extremely slow(or I dont know what is the relatively good speed)
            Batch size 128, training time is around 8s


    :return:
    """
    print("loading model from generator ")
    # tr, va, te = loads()
    loader = Minc2500()

    tr_iterator = loader.generator(input_file='test1.txt', target_size=(INPUT_SHAPE[1], INPUT_SHAPE[2]))
    # x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_alex_original()
    x, weight_path, img_input = create_VGG_snd()
    model = Model(input=img_input,
                  output=[x])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Load the weights
    model.load_weights(weight_path, by_name=True)

    model.fit_generator(tr_iterator, samples_per_epoch=128*100, nb_epoch=NB_EPOCH, nb_worker=4)

    # score = model.evaluate(te[0], te[1], verbose=0)
    # print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))


def test_minc_original_VGG_generator():
    """
    This is testing case for minc dataset on original Alexnet as entry-point
        Experiment result:
            Not converging from random-initialization
            Extremely slow(or I dont know what is the relatively good speed)
            Batch size 128, training time is around 8s


    :return:
    """
    print("loading model from generator ")
    # tr, va, te = loads()
    loader = Minc2500()

    tr_iterator = loader.generator(input_file='train1.txt', target_size=(INPUT_SHAPE[1], INPUT_SHAPE[2]))
    te_iterator = loader.generator(input_file='test1.txt', target_size=(INPUT_SHAPE[1], INPUT_SHAPE[2]))

    tr_sample = tr_iterator.nb_sample
    te_sample = te_iterator.nb_sample

    # x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_alex_original()
    x, weight_path, img_input = create_VGG_original2()
    model = Model(input=img_input,
                  output=[x])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Load the weights
    model.load_weights(weight_path, by_name=True)

    model.fit_generator(tr_iterator, samples_per_epoch=128*100, nb_epoch=NB_EPOCH, nb_worker=4,
                        validation_data=te_iterator, nb_val_samples=te_sample)

    # score = model.evaluate(te[0], te[1], verbose=0)
    # print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))z



# def test_train_generator():
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


if __name__ == '__main__':
    test_minc_original_VGG_generator()
    # test_minc_original_alexnet_reduced()
    # test_VGG()
    # check_print('original')
    # check_print('second')