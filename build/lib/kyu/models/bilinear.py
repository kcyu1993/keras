
from keras.applications import VGG16, ResNet50
from keras.layers import Conv2D, Dense
from kyu.engine.configs import ModelConfig
from kyu.models.secondstat import BiLinear
from kyu.theano.general.train import toggle_trainable_layers, Model


def _compose_bilinear_model(base_model, nb_class, freeze_conv=False, name='Bilinear_default'):
    # Create Dense layers
    x = base_model.output
    x = Conv2D(256, (1, 1), name='1x1_stage5')(x)
    x = BiLinear(eps=0, activation='linear')(x)
    x = Dense(nb_class, activation='softmax')(x)
    if freeze_conv:
        toggle_trainable_layers(base_model, trainable=False)

    new_model = Model(base_model.input, x, name=name)
    return new_model


def VGG16_bilinear(nb_class, load_weights='imagenet', input_shape=(224,224,3), freeze_conv=False):
    if load_weights == 'imagenet':
        base_model = VGG16(include_top=False, input_shape=input_shape)
    elif load_weights is None:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
    else:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_bilinear_model(base_model=base_model, nb_class=nb_class,
                                   freeze_conv=freeze_conv, name='VGG16_bilinear')


def ResNet50_bilinear(nb_class, load_weights='imagenet', input_shape=(224,224,3), last_avg=False, freeze_conv=False):
    if load_weights == 'imagenet':
        base_model = ResNet50(include_top=False, input_shape=input_shape, last_avg=last_avg)
    elif load_weights is None:
        base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
    else:
        base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_bilinear_model(base_model=base_model, nb_class=nb_class,
                                   freeze_conv=freeze_conv, name='ResNet50_bilinear')


class BilinearConfig(ModelConfig):

    def __init__(self,
                 nb_class,
                 input_shape,
                 class_id='vgg',
                 model_id='bilinear',
                 load_weights='imagenet'
                 ):
        super(BilinearConfig, self).__init__(class_id, model_id)
        self.__dict__.update(locals())
