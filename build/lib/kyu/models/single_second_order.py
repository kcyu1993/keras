"""
Define single stream SO-CNN for both ResNet and VGG and others with wrapper.

"""

import keras.backend as K
from keras.applications import VGG16
from keras.layers import Flatten, Dense, merge, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import add, average, concatenate
from keras.models import Model
from kyu.layers.secondstat import WeightedVectorization
from kyu.layers.assistants import FlattenSymmetric, SeparateConvolutionFeatures, MatrixConcat
from kyu.models.alexnet5 import AlexNet_v2
from kyu.models.densenet121 import DenseNet121
from kyu.models.resnet50 import ResNet50_v2
from kyu.models.so_cnn_helper import get_cov_block, upsample_wrapper_v1
from kyu.utils.dict_utils import merge_dicts
from kyu.utils.train_utils import toggle_trainable_layers


def _compose_second_order_model(
        base_model, nb_class, cov_branch, cov_branch_kwargs=None,
        ######## FOR DCovConfig #########
        mode=0,
        cov_branch_output=None,
        dense_branch_output=None,
        freeze_conv=False, name='default_so_model',
        nb_branch=1,
        nb_outputs=1,
        concat='concat',
        cov_output_vectorization='pv',
        last_conv_feature_maps=[],
        last_conv_kernel=[1,1],
        upsample_method='conv',
        # pass to the cov_branch_fn
        **kwargs
):
    if nb_outputs > 1:
        raise ValueError("Compose second order doesn't support nb_outputs larger than 1")
    cov_branch_fn = get_cov_block(cov_branch)
    if cov_branch_output is None:
        cov_branch_output = nb_class
    if dense_branch_output is None:
        dense_branch_output = nb_class
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
            # Repeat the traditional VGG16.
            # fo = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(base_model.output)
            # fo = GlobalAveragePooling2D()(fo)
            fo = Flatten()(x)
            fo = Dense(dense_branch_output, activation='relu')(fo)
            cov_branch_y = merge([fo, cov_branch_y], mode=concat)
        else:
            raise ValueError("Not supported mode {}".format(mode))
        cov_outputs.append(cov_branch_y)

    # Merge if valid
    if nb_branch > 1:
        if concat == 'concat':
            if all(len(K.int_shape(cov_output)) == 3 for cov_output in cov_outputs):
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

    if len(K.int_shape(x)) == 3:
        if cov_output_vectorization == 'pv' or cov_output_vectorization == 'wv':
            if cov_branch_kwargs.has_key('pv_kwargs'):
                pv_kwargs = cov_branch_kwargs.get('pv_kwargs')
            else:
                pv_kwargs = {}
            x = WeightedVectorization(cov_branch_output, **pv_kwargs)(x)
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
    if load_weights in {"imagenet", "secondorder"}:
        base_model = VGG16(include_top=False, input_shape=input_shape, pooling=pooling,
                           weights=load_weights)
    elif load_weights is None:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
    else:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_second_order_model(base_model, nb_class, cov_branch, **kwargs)


def DenseNet121_second_order(
        input_shape, nb_class, cov_branch, load_weights='imagenet', pooling=None, **kwargs

):
    if load_weights in {"imagenet", "secondorder"}:
        base_model = DenseNet121(include_top=False, input_shape=input_shape, last_pooling=pooling,
                                 weights_path='imagenet')
    elif load_weights is None:
        base_model = DenseNet121(include_top=False, weights_path=None, input_shape=input_shape, last_pooling=pooling)
    else:
        base_model = DenseNet121(include_top=False, weights_path=None, input_shape=input_shape, last_pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_second_order_model(base_model, nb_class, cov_branch, **kwargs)


def ResNet50_second_order(
        input_shape, nb_class, cov_branch, load_weights='imagenet',
        pooling=None, last_avg=False,
        **kwargs
):

    if load_weights in {"imagenet", "secondorder"}:
        base_model = ResNet50_v2(include_top=False, input_shape=input_shape, last_avg=last_avg, pooling=pooling,
                                 weights=load_weights)
    elif load_weights is None:
        base_model = ResNet50_v2(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg,
                                 pooling=pooling)
    else:
        base_model = ResNet50_v2(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg,
                                 pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_second_order_model(base_model, nb_class, cov_branch, **kwargs)


def AlexNet_second_order(
        input_shape, nb_class, cov_branch, load_weights='imagenet', pooling='max', **kwargs
):
    if load_weights in {"imagenet", "secondorder"}:
        base_model = AlexNet_v2(include_top=False, input_shape=input_shape, pooling=pooling)
    elif load_weights is None:
        base_model = AlexNet_v2(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
    else:
        base_model = AlexNet_v2(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_second_order_model(base_model, nb_class, cov_branch, **kwargs)
