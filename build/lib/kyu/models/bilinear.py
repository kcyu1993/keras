
from keras.applications import VGG16, ResNet50
from keras.layers import Dense, Conv2D, BatchNormalization
from keras.models import Model
from kyu.layers.secondstat import BiLinear
from kyu.layers.assistants import FlattenSymmetric, SignedSqrt, L2Norm
from kyu.models.densenet121 import DenseNet121
from kyu.utils.train_utils import toggle_trainable_layers


def _compose_bilinear_model(base_model, nb_class, freeze_conv=False, last_conv_kernel=[], name='Bilinear_default'):
    # Create Dense layers
    x = base_model.output
    for k in last_conv_kernel:
        x = Conv2D(k, (1, 1), name='1x1_stage5_{}'.format(k))(x)

    # x = BatchNormalization(axis=3, name='last_BN')(x)
    x = BiLinear(eps=0, activation='linear')(x)
    x = FlattenSymmetric()(x)
    x = SignedSqrt()(x)
    x = L2Norm(axis=1)(x)
    x = Dense(nb_class, activation='softmax')(x)
    if freeze_conv:
        toggle_trainable_layers(base_model, trainable=False)

    new_model = Model(base_model.input, x, name=name)
    return new_model


def DenseNet121_bilinear(nb_class, load_weights='imagenet', input_shape=(224,224,3), freeze_conv=False, **kwargs):
    if load_weights in {"imagenet", "secondorder"}:
        base_model = DenseNet121(include_top=False, input_shape=input_shape)
    elif load_weights is None:
        base_model = DenseNet121(include_top=False, weights_path=None, input_shape=input_shape)
    else:
        base_model = DenseNet121(include_top=False, weights_path=None, input_shape=input_shape)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_bilinear_model(base_model=base_model, nb_class=nb_class,
                                   freeze_conv=freeze_conv, name='DenseNet121_bilinear',
                                   **kwargs)


def VGG16_bilinear(nb_class, load_weights='imagenet', input_shape=(224,224,3), freeze_conv=False,
                   **kwargs):
    if load_weights in {"imagenet", "secondorder"}:
        base_model = VGG16(include_top=False, input_shape=input_shape)
    elif load_weights is None:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
    else:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_bilinear_model(base_model=base_model, nb_class=nb_class,
                                   freeze_conv=freeze_conv, name='VGG16_bilinear',
                                   **kwargs)


def ResNet50_bilinear(nb_class, load_weights='imagenet', input_shape=(224,224,3), last_avg=False, freeze_conv=False,
                      **kwargs):
    if load_weights in {"imagenet", "secondorder"}:
        base_model = ResNet50(include_top=False, input_shape=input_shape, last_avg=last_avg)
    elif load_weights is None:
        base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
    else:
        base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_bilinear_model(base_model=base_model, nb_class=nb_class,
                                   freeze_conv=freeze_conv, name='ResNet50_bilinear',
                                   **kwargs)


