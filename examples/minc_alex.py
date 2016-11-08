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
from keras.layers.custom_layers import LRN2D
from keras.datasets.minc import Minc2500
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



def conv2D_lrn2d(x, nb_filter, nb_row, nb_col,
                 border_mode='same', subsample=(1, 1),
                 activation='relu', LRN2D_norm=True,
                 weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''

        Info:
            Function taken from the Inceptionv3.py script keras github


            Utility function to apply to a tensor a module Convolution + lrn2d
            with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      bias=False,
                      dim_ordering=dim_ordering)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    if LRN2D_norm:

        x = LRN2D(alpha=ALPHA, beta=BETA)(x)
        x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    return x


def create_model_alex():
    # Define image input layer
    if DIM_ORDERING == 'th':
        INP_SHAPE = (3, 362, 362)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        INP_SHAPE = (362, 362, 3)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Channel 1 - Convolution Net Layer 1
    x = conv2D_lrn2d(
        img_input, 3, 11, 11, subsample=(
            1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 2
    x = conv2D_lrn2d(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 3
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 4
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 5
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Cov Net Layer 6
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # # Channel 1 - Cov Net Layer 7
    # x = Dense(2048, activation='relu')(x)
    # x = Dropout(DROPOUT)(x)
    #
    # # Channel 1 - Cov Net Layer 8
    # x = Dense(2048, activation='relu')(x)
    # x = Dropout(DROPOUT)(x)
    #
    return x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING

# def minc_model1():
#     model = Sequential()
#     model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                             border_mode='valid',
#                             input_shape=input_shape,
#                             activation='relu'))
#     model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                             activation='relu'))
#     model.add(MaxPooling2D(pool_size=pool_size))
#     model.add(Dropout(0.25))
#
#     # Start of secondary layer
#     model.add(Flatten())
#
#     model.add(Dense(nb_classes))
#     # model.add(Activation('relu'))
#     model.add(Activation('softmax'))
#
#     opt = optimizers.rmsprop()
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#     return model

def create_alex_snd():
    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model_alex()
    x = SecondaryStatistic(output_dim=None, parametrized=False, init='normal')(x)
    x = WeightedProbability(output_dim=NB_CLASS, activation='softmax')(x)
    return x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING

def create_alex_original():
    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model_alex()
    # Final Channel - Cov Net 9
    x = Flatten()(x)
    x = Dense(output_dim=NB_CLASS,
              activation='softmax')(x)
    return x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING


def check_print(name='second'):
    # Create the Model
    if name is 'second':
        x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_alex_snd()
    elif name is 'original':
        x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_alex_original()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[x])
    model.summary()

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


def test_minc_original_alexnet_reduced():
    print("load reduced model from raw image data")
    loader = Minc2500()

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = loader.loadwithsplit()
    tr_Y = to_categorical(tr_y, nb_classes=NB_CLASS)
    va_Y = to_categorical(va_y, nb_classes=NB_CLASS)
    te_Y = to_categorical(te_y, nb_classes=NB_CLASS)

    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_alex_original()
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

def test_minc_original_alexnet_generator():
    print("loading model from generator ")
    # tr, va, te = load()
    loader = Minc2500()

    tr_iterator = loader.generator(file='test1.txt')
    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_alex_original()
    model = Model(input=img_input,
                  output=[x])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(tr_iterator, samples_per_epoch=40000, nb_epoch=NB_EPOCH, nb_worker=4)

    # score = model.evaluate(te[0], te[1], verbose=0)
    # print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))


# def test_train_generator():



if __name__ == '__main__':
    test_minc_original_alexnet_generator()
    # test_minc_original_alexnet_reduced()
    # check_print('original')
    # check_print('second')