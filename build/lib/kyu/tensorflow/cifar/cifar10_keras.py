from kyu.legacy.so_cnn_helper import covariance_block_vector_space
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import Convolution2D, MaxPooling2D, O2Transform, LRN2D
from keras.layers import Dense, Dropout, Activation, Flatten, SecondaryStatistic, WeightedVectorization
from keras.models import Sequential
import keras
import tensorflow as tf

if keras.backend._BACKEND == 'tensorflow':
    keras.backend.set_image_dim_ordering('tf')
    keras.backend.set_learning_phase(1)
else:
    keras.backend.set_image_dim_ordering('th')

nb_classes = 10
input_shape = (32, 32, 3)

def comparison_model(input_tensor):
    pass


def cifar10_tensorflow_model():
    img_input = Input(input_shape, name="orig_input")
    x = Convolution2D(64, 5, 5, init='normal', border_mode='same', name='conv1')(img_input)
    x = MaxPooling2D((3,3), strides=(2,2), name='pool1')(x)
    x = LRN2D(n=4, name='norm1')(x)


def cifar_fitnet_v1_test(input_tensor, init='glorot_normal'):
    x = Convolution2D(16, 3, 3, init=init, border_mode='same')(input_tensor)
    x = Activation('relu')(x)
    x = Convolution2D(16,3,3, init=init, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(16, 3, 3, init=init, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(32, 3, 3, init=init, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, 3, 3, init=init, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, 3, 3, init=init, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Convolution2D(48, 3, 3, init=init, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(48, 3, 3, init=init, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 3, 3, init=init, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    # TODO Check flatten
    x = Dense(500, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    return x
    pass

def cifar_fitnet_v1(input_shape):
    """
    Implement the fit model has 205K param
    Without any Maxout design in this version
    Just follows the general architecture

    :return: model sequential
    """

    basename = 'fitnet_v1'
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(48, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Dense(nb_classes, activation='softmax'))
    basename += "_fc"

    model.name = basename
    return model


def cifar_fitnet_v2(parametrics=[], epsilon=0., mode=0):
    """
        Implement the fit model has 205K param
        Without any Maxout design in this version
        Just follows the general architecture

        :return: model sequential
        """
    nb_class = nb_classes
    basename = 'fitnet_v2'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}'.format(str(mode))

    if epsilon > 0:
        basename += '-epsilon_{}'.format(str(epsilon))

    input_tensor = Input(input_shape)
    x = Convolution2D(16, 3, 3, border_mode='same')(input_tensor)
    x = Activation('relu')(x)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    block1_x = x

    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    block2_x = x
    x = Convolution2D(48, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(48, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    block3_x = x

    cov_input = block3_x
    if mode == 0: # Original Network
        x = Flatten()(x)
        x = Dense(500)(x)
        x = Dense(nb_classes)(x)
        x = Activation('softmax')(x)
    elif mode == 1: # Original Cov_Net
        x = covariance_block_vector_space(x, nb_class, stage=4, block='a', epsilon=epsilon, parametric=parametrics)
        x = Activation('softmax')(x)

    elif mode == 2: # Concat balanced
        cov_branch = covariance_block_vector_space(cov_input, nb_class,
                                                   stage=4, epsilon=epsilon,
                                                   block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 3: # Concat two softmax
        cov_branch = covariance_block_vector_space(cov_input, nb_class, epsilon=epsilon,
                                                   stage=4, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc_softmax')(x)
        cov_branch = Activation('softmax')(cov_branch)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 4: # Concat multiple branches (balanced)
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, epsilon=epsilon,
                                                    stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, epsilon=epsilon,
                                                    stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, epsilon=epsilon,
                                                    stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class)(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 5: # Concat multiple 'softmax' final layers
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, epsilon=epsilon,
                                                    stage=2, block='a',
                                                    parametric=parametrics, activation='softmax')
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, epsilon=epsilon,
                                                    stage=3, block='b',
                                                    parametric=parametrics, activation='softmax')
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, epsilon=epsilon,
                                                    stage=4, block='c',
                                                    parametric=parametrics, activation='softmax')
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc_softmax')(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 6: # Average multiple relu
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, epsilon=epsilon,
                                                    stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, epsilon=epsilon,
                                                    stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, epsilon=epsilon,
                                                    stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        cov_branch = merge([cov_branch1, cov_branch2, cov_branch3], mode='ave', name='average')
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 7: # Average multiple softmax
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, epsilon=epsilon,
                                                    stage=2, block='a',
                                                    parametric=parametrics, activation='softmax')
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, epsilon=epsilon,
                                                    stage=3, block='b',
                                                    parametric=parametrics, activation='softmax')
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, epsilon=epsilon,
                                                    stage=4, block='c',
                                                    parametric=parametrics, activation='softmax')
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc_softmax')(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='ave', name='average')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    else:
        raise ValueError("Mode not supported {}".format(mode))

    model = Model(input_tensor, x, name=basename)
    return model