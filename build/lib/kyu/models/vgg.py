"""
Re implement VGG model for general usage

"""
import six

# KERAS Import
import keras.backend as K
from keras import Input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.vgg16 import WEIGHTS_PATH, WEIGHTS_PATH_NO_TOP
from keras.engine import Model, get_source_inputs
from keras.layers import Flatten, Dense, warnings, Conv2D, MaxPooling2D
from keras.utils import get_file, layer_utils
from kyu.configs.engine_configs import ModelConfig
from kyu.configs.model_configs.second_order import DCovConfig
from kyu.legacy.vgg16 import VGG16_o2
from kyu.models.bilinear import VGG16_bilinear
from kyu.models.single_second_order import VGG16_second_order
from kyu.utils.train_utils import toggle_trainable_layers
from .generic_loader import deserialize_model_object, get_model_from_config

VGG_SUPPORTED_MODEL = ['vgg16',]


def VGG16_first_order(
        denses=[], nb_class=1000, input_shape=None,
        input_tensor=None,
        weights='imagenet',
        include_top=True,
        freeze_conv=False,
        last_pooling=True):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    if last_pooling:
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        if nb_class != 1000:
            pred_name = 'new_pred'
        else:
            pred_name = 'predictions'
        base_model = Model(inputs, x, name='vgg16_withtop')
        toggle_trainable_layers(base_model, not freeze_conv)
    else:
        base_model = Model(inputs, x, name='vgg16_notop')
        toggle_trainable_layers(base_model, not freeze_conv)
        x = Flatten(name='flatten')(x)
        for ind, para in enumerate(denses):
            x = Dense(para, activation='relu', name='new_fc{}'.format(str(ind+1)),
                      kernel_initializer='glorot_uniform')(x)
        pred_name = 'new_pred'

    # x = base_model.output
    x = Dense(nb_class, activation='softmax', name=pred_name)(x)

    # Create model.
    model = Model(base_model.inputs, x, base_model.name)

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        print('{} load weights from {} by name.'.format(model.name, weights_path))
        model.load_weights(weights_path, by_name=True)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def second_order(config):
    """ Implement the original SO-VGG16 with single stream structure """
    if not isinstance(config, DCovConfig):
        raise ValueError("VGG: second-order only support DCovConfig")

    compulsory = DCovConfig.compulsoryArgs
    optional = DCovConfig.optionalArgs
    return get_model_from_config(VGG16_second_order, config, compulsory, optional)


def multiple_loss_second_order(config):
    if not isinstance(config, DCovConfig) and hasattr(config, 'nb_outputs'):
        raise ValueError("VGG: second-order only support DCovConfig has nb_outputs")
    if not config.nb_outputs > 1:
        raise ValueError("VGG: multiple_loss_second_order supports only nb_outputs > 1")
    compulsory = DCovConfig.compulsoryArgs
    optional = DCovConfig.optionalArgs

    return get_model_from_config(VGG16_second_order, config, compulsory, optional)


def bilinear(config):
    """ Create the bilinear model and return """
    if not isinstance(config, ModelConfig):
        raise ValueError("VGG: get_model only support ModelConfig object")

    compulsory = ['nb_class', 'input_shape']
    optional = ['load_weights', 'input_shape', 'freeze_conv', 'last_conv_kernel']

    # Return the model
    # return VGG16_bilinear(**args)
    return get_model_from_config(VGG16_bilinear, config, compulsory, optional)


def first_order(config):
    """ Create for first order VGG """
    if not isinstance(config, ModelConfig):
        raise ValueError("VGG_first_order: only support First_order_Config")
    compulsory = ['nb_class', 'input_shape']
    optional = ['denses', 'input_tensor', 'weights', 'include_top', 'last_pooling', 'freeze_conv']
    return get_model_from_config(VGG16_first_order, config, compulsory, optional)


def deserialize(name, custom_objects=None):
    return deserialize_model_object(name,
                                    module_objects=globals(),
                                    printable_module_name='vgg model'
                                    )


def get_model(config):
    if not isinstance(config, ModelConfig):
        raise ValueError("VGG: get_model only support ModelConfig object")

    model_id = config.model_id
    # Decompose the config with different files.
    # if model_id == ''
    if isinstance(model_id, six.string_types):
        model_id = str(model_id)
        model_function = deserialize(model_id)
    elif callable(model_id):
        model_function = model_id
    else:
        raise ValueError("Could not interpret the vgg model with {}".format(model_id))

    return model_function(config)


if __name__ == '__main__':
    # model = VGG16_o1([4096,4096,4096], input_shape=(224,224,3))
    model = VGG16_o2([256, 256, 128], nb_branch=2, cov_mode='pmean', freeze_conv=True,
                     cov_branch_output=64,
                     robust=True, mode=1, nb_classes=47)
    model.compile('sgd', 'categorical_crossentropy')
    model.summary()
