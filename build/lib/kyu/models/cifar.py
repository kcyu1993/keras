from __future__ import absolute_import

from keras.applications.resnet50 import covariance_block_original, ResNet50CIFAR
from keras.engine import Input
from keras.engine import merge
from keras.layers import \
    SecondaryStatistic, Convolution2D, Activation, MaxPooling2D, Dropout, \
    Flatten, O2Transform, Dense, WeightedProbability
from keras.models import Sequential, Model


def cifar_fitnet_v1(second=False, parametric=[], nb_classes=10, input_shape=(3,32,32)):
    """
    Implement the fit model has 205K param
    Without any Maxout design in this version
    Just follows the general architecture

    :return: model sequential
    """
    basename = 'fitnet_v1'

    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=input_shape))
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

    if not second:
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Dense(nb_classes, activation='softmax'))
        basename += "_fc"
    else:
        model.add(SecondaryStatistic(name='second_layer'))
        basename += "_snd"
        if parametric is not []:
            basename += '_para-'
            for para in parametric:
                basename += str(para) + '_'
        for ind, para in enumerate(parametric):
            model.add(O2Transform(output_dim=para, name='O2transform_{}'.format(ind)))
        model.add(WeightedProbability(output_dim=nb_classes))
        model.add(Activation('softmax'))

    model.name = basename
    return model


def cifar_fitnet_v2(parametrics=[], epsilon=0., mode=0, nb_classes=10, input_shape=(3,32,32)):
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
        x = covariance_block_original(x, nb_class, stage=4, block='a', epsilon=epsilon, parametric=parametrics)
        x = Activation('softmax')(x)

    elif mode == 2: # Concat balanced
        cov_branch = covariance_block_original(cov_input, nb_class,
                                               stage=4, epsilon=epsilon,
                                               block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 3: # Concat two softmax
        cov_branch = covariance_block_original(cov_input, nb_class, epsilon=epsilon,
                                               stage=4, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc_softmax')(x)
        cov_branch = Activation('softmax')(cov_branch)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 4: # Concat multiple branches (balanced)
        cov_branch1 = covariance_block_original(block1_x, nb_class, epsilon=epsilon,
                                                stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_original(block2_x, nb_class, epsilon=epsilon,
                                                stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_original(block3_x, nb_class, epsilon=epsilon,
                                                stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class)(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 5: # Concat multiple 'softmax' final layers
        cov_branch1 = covariance_block_original(block1_x, nb_class, epsilon=epsilon,
                                                stage=2, block='a',
                                                parametric=parametrics, activation='softmax')
        cov_branch2 = covariance_block_original(block2_x, nb_class, epsilon=epsilon,
                                                stage=3, block='b',
                                                parametric=parametrics, activation='softmax')
        cov_branch3 = covariance_block_original(block3_x, nb_class, epsilon=epsilon,
                                                stage=4, block='c',
                                                parametric=parametrics, activation='softmax')
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc_softmax')(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 6: # Average multiple relu
        cov_branch1 = covariance_block_original(block1_x, nb_class, epsilon=epsilon,
                                                stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_original(block2_x, nb_class, epsilon=epsilon,
                                                stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_original(block3_x, nb_class, epsilon=epsilon,
                                                stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        cov_branch = merge([cov_branch1, cov_branch2, cov_branch3], mode='ave', name='average')
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 7: # Average multiple softmax
        cov_branch1 = covariance_block_original(block1_x, nb_class, epsilon=epsilon,
                                                stage=2, block='a',
                                                parametric=parametrics, activation='softmax')
        cov_branch2 = covariance_block_original(block2_x, nb_class, epsilon=epsilon,
                                                stage=3, block='b',
                                                parametric=parametrics, activation='softmax')
        cov_branch3 = covariance_block_original(block3_x, nb_class, epsilon=epsilon,
                                                stage=4, block='c',
                                                parametric=parametrics, activation='softmax')
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc_softmax')(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='ave', name='average')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 8:
        block1_x = Flatten()(block1_x)
        block2_x = Flatten()(block2_x)
        # block3_x = Flatten()(block3_x)
        dense_branch1 = Dense(nb_class, activation='relu', name='fc_block1')(block1_x)
        dense_branch2 = Dense(nb_class, activation='relu', name='fc_block2')(block2_x)
        # dense_branch3 = Dense(nb_class, activation='relu', name='fc_block3')(block3_x)

        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc_relu')(x)
        # x = merge([x, dense_branch1, dense_branch2], mode='sum', name='sum')
        x = merge([x, dense_branch1, dense_branch2], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    else:
        raise ValueError("Mode not supported {}".format(mode))

    model = Model(input_tensor, x, name=basename)
    return model


def model_original(nb_classes=10, input_shape=(3,32,32)):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


def model_snd(parametric=True, input_shape=(3,32,32)):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(SecondaryStatistic(activation='linear'))
    if parametric:
        model.add(O2Transform(activation='relu', output_dim=100))
    model.add(WeightedProbability(10,activation='linear', init='normal'))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    return model


def resnet50_original(nb_classes=10):
    # img_input = Input(shape=(img_channels, img_rows, img_cols))
    # model = ResNet50(True, weights='imagenet')
    model = ResNet50CIFAR(nb_class=nb_classes)
    # let's train the model using SGD + momentum (how original).
    return model


def resnet50_snd(parametric=False, nb_classes=10):
    x, img_input = ResNet50CIFAR(False, nb_class=nb_classes)
    x = SecondaryStatistic(parametrized=False)(x)
    if parametric:
        x = O2Transform(output_dim=100, activation='relu')(x)
        x = O2Transform(output_dim=10, activation='relu')(x)
    x = WeightedProbability(output_dim=nb_classes, activation='softmax')(x)
    model = Model(img_input, x, name='ResNet50CIFAR_snd')
    return model
