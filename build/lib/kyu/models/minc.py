
from keras.applications import VGG19
from keras.applications.resnet50 import ResCovNet50MINC, ResNet50MINC, covariance_block_vector_space
from keras.applications.vgg19 import VGG19_bottom
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import O2Transform, WeightedProbability, Dense, Flatten, Activation, Convolution2D, MaxPooling2D, \
    Dropout
from keras.layers import SecondaryStatistic

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
INPUT_SHAPE = (3, 224, 224)



def minc_fitnet_v2(parametrics=[], epsilon=0., mode=0, nb_classes=23, input_shape=(3,224,224)):
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
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    x = MaxPooling2D(pool_size=(2, 2))(x)

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

    elif mode == 8:
        block1_x = Flatten()(block1_x)
        block2_x = Flatten()(block2_x)
        # block3_x = Flatten()(block3_x)
        dense_branch1 = Dense(nb_class, activation='relu', name='fc_block1')(block1_x)
        dense_branch2 = Dense(nb_class, activation='relu', name='fc_block2')(block2_x)
        # dense_branch3 = Dense(nb_class, activation='relu', name='fc_block3')(block3_x)

        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc_relu')(x)
        # x = merge([x, dense_branch1, dense_branch2], cov_mode='sum', name='sum')
        x = merge([x, dense_branch1, dense_branch2], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    else:
        raise ValueError("Mode not supported {}".format(mode))

    model = Model(input_tensor, x, name=basename)
    return model



def create_ResCovNet50(para=[], start=0, stop=0, verbose=2):
    model_list = []
    for mode in range(start, stop):
        model = ResCovNet50MINC(parametrics=para, mode=mode)
        model_list.append(model)
    return model_list


def create_ResNet50(second=False, parametric=True):
    if not second:
        model = ResNet50MINC(weights='imagenet', nb_class=NB_CLASS)
        model.name = 'ResNet50_original'
    else:
        x, img_input, weight_path = ResNet50MINC(include_top=False, nb_class=NB_CLASS, weights='imagenet')
        x = SecondaryStatistic()(x)
        if parametric:
            x = O2Transform(output_dim=100, activation='relu')(x)
        x = WeightedProbability(output_dim=NB_CLASS, activation='softmax')(x)
        model = Model(img_input, x)
        model.name = 'ResNet_second'
    return model

def create_VGG_snd():
    x, weight_path, img_input = VGG19_bottom(include_top=False, weights='imagenet')
    x = SecondaryStatistic(output_dim=None, parametrized=False, init='normal')(x)
    x = WeightedProbability(output_dim=NB_CLASS, activation='softmax')(x)
    model = Model(img_input, x)
    return model


def create_VGG_original2():
    x, weight_path, img_input = VGG19_bottom(include_top=True, weights='imagenet', output_dim=NB_CLASS)
    return Model(img_input, x)
    # return x, weight_path, img_input


def create_alex_original():
    return VGG19(weights='imagenet')