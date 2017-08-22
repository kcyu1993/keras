"""
Define single stream SO-CNN for both ResNet and VGG and others with wrapper.

"""
from keras.applications import VGG16, ResNet50
from keras.layers import Flatten, Dense, merge
from keras.layers.merge import add, average, concatenate
from kyu.engine.configs import ModelConfig
from kyu.models.secondstat import SeparateConvolutionFeatures, MatrixConcat, WeightedVectorization, FlattenSymmetric
from kyu.theano.general.train import toggle_trainable_layers, Model
from kyu.utils.sys_utils import merge_dicts
from .so_cnn_helper import get_cov_block, upsample_wrapper_v1


def _compose_second_order_model(
        base_model, nb_class, cov_branch, cov_branch_kwargs=None,

        ######## FOR DCovConfig #########
        mode=0, cov_branch_output=None,
        freeze_conv=False, name='default_so_model',
        nb_branch=1,
        concat='concat',
        cov_output_vectorization='pv',
        last_conv_feature_maps=[],
        last_conv_kernel=[1,1],
        upsample_method='conv',

        **kwargs # pass to the cov_branch_fn
):
    cov_branch_fn = get_cov_block(cov_branch)
    if cov_branch_output is None:
        cov_branch_output = nb_class

    x = base_model.output

    if freeze_conv:
        toggle_trainable_layers(base_model, False)

    x = upsample_wrapper_v1(x, last_conv_feature_maps, upsample_method, kernel=last_conv_kernel)

    if nb_branch > 1:
        cov_input = SeparateConvolutionFeatures(nb_branch)(x)
    else:
        cov_input = [x]

    cov_outputs = []
    for ind, x in enumerate(cov_input):
        if mode == 1:
            cov_branch_y = cov_branch_fn(x, cov_branch_output, stage=5, block=str(ind),
                                         ** merge_dicts(kwargs, cov_branch_kwargs))
        elif mode == 2:
            cov_branch_y = cov_branch_fn(x, cov_branch_output, stage=5, block=str(ind),
                                         ** merge_dicts(kwargs, cov_branch_kwargs))
            fo = Flatten()(x)
            fo = Dense(nb_class, activation='relu')(fo)
            cov_branch_y = merge([fo, cov_branch_y], mode=concat)
        else:
            raise ValueError("Not supported mode {}".format(mode))
        cov_outputs.append(cov_branch_y)

    # Merge if valid
    if nb_branch > 1:
        if concat == 'concat':
            if all(len(cov_output.shape) == 3 for cov_output in cov_outputs):
                # use matrix concat
                x = MatrixConcat(cov_outputs)(cov_outputs)
            else:
                x = concatenate(cov_outputs)
        elif concat == 'sum':
            x = add(cov_outputs)
        elif concat == 'ave' or concat == 'avg':
            x = average(cov_outputs)
        else:
            raise ValueError("Concat mode not supported {}".format(concat))
    else:
        x = cov_outputs[0]

    if len(x.shape) == 3:
        if cov_output_vectorization == 'pv' or cov_output_vectorization == 'wv':
            x = WeightedVectorization(cov_branch_output, activation='relu', use_bias=False)(x)
        elif cov_output_vectorization == 'mat_flatten':
            x = FlattenSymmetric()(x)
        else:
            x = Flatten()(x)

    x = Dense(nb_class, activation='softmax')(x)

    model = Model(base_model.input, x, name=name)
    return model


def VGG16_second_order(
        input_shape, nb_class, cov_branch, load_weights='imagenet', pooling=None, **kwargs

):
    if load_weights == 'imagenet':
        base_model = VGG16(include_top=False, input_shape=input_shape, pooling=pooling)
    elif load_weights is None:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
    else:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_second_order_model(base_model, nb_class, cov_branch, **kwargs)


def ResNet50_second_order(
        input_shape, nb_class, cov_branch, load_weights='imagenet', pooling=None, last_avg=False, **kwargs
):
    if load_weights == 'imagenet':
        base_model = ResNet50(include_top=False, input_shape=input_shape, last_avg=last_avg, pooling=pooling)
    elif load_weights is None:
        base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg,
                              pooling=pooling)
    else:
        base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg,
                              pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_second_order_model(base_model, nb_class, cov_branch, **kwargs)


class DCovConfig(ModelConfig):

    def __init__(self,
                 input_shape,
                 nb_class,
                 cov_branch,
                 cov_branch_kwargs,
                 class_id='vgg',
                 load_weights='imagenet',

                 # configs for _compose_second_order_things
                 mode=0, cov_branch_output=None,
                 freeze_conv=False, name='default_so_model',
                 nb_branch=1,
                 concat='concat',
                 cov_output_vectorization='pv',
                 last_conv_feature_maps=[],
                 last_conv_kernel=[1, 1],
                 upsample_method='conv',
                 **kwargs
                 ):
        model_id = 'second_order'
        super(DCovConfig, self).__init__(class_id, model_id)
        self.__dict__.update(locals())


class O2TBranchConfig(DCovConfig):
    def __init__(self,
                 parametric=[],
                 activation='relu',
                 cov_mode='channel',
                 vectorization='wv',
                 epsilon=1e-5,
                 use_bias=True,
                 **kwargs
                 ):
        z = {}
        z['parametric'] = parametric
        z['activation'] = activation
        z['cov_mode'] = cov_mode
        z['vectorization'] = vectorization
        z['epsilon'] = epsilon
        z['use_bias'] = use_bias
        super(O2TBranchConfig, self).__init__(cov_branch_kwargs=z, **kwargs)


class NoWVBranchConfig(DCovConfig):
    def __init__(self,
                 parametric=[],
                 epsilon=1e-7,
                 activation='relu',
                 **kwargs
                 ):
        z = merge_dicts(locals(), kwargs)
        super(NoWVBranchConfig, self).__init__(**kwargs)