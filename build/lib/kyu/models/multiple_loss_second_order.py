"""
Implement the multiple loss models (actually,
make single_second_order a special case of multiple loss)

"""


"""
Define single stream SO-CNN for both ResNet and VGG and others with wrapper.

"""

import keras.layers.merge as merge
from keras.layers import Dense
from keras.models import Model
from kyu.utils.dict_utils import merge_dicts
from kyu.utils.train_utils import toggle_trainable_layers
from .so_cnn_helper import get_cov_block, upsample_wrapper_v1, merge_branches_with_method


def _compose_general_second_order_model(
        base_model, nb_class, cov_branch, cov_branch_kwargs=None,
        ######## FOR DCovConfig #########
        mode=0, cov_branch_output=None,
        freeze_conv=False, name='default_so_general_model',
        nb_branch=1,
        concat='concat',
        sub_branch_concat='sum',
        cov_output_vectorization='pv',
        last_conv_feature_maps=[],
        last_conv_kernel=[1,1],
        upsample_method='conv',
        # pass to the cov_branch_fn
        **kwargs
):
    if mode <= 2:
        raise ValueError("General second order model now only support mode > 2")

    cov_branch_fn = get_cov_block(cov_branch)
    if cov_branch_output is None:
        cov_branch_output = nb_class

    base_outputs = base_model.outputs
    # x = base_model.output

    # Toggle layers
    if freeze_conv:
        toggle_trainable_layers(base_model, False)

    cov_inputs = base_outputs
    cov_outputs = []

    for ind, x in enumerate(cov_inputs):
        cov_output = []

        # Add 1x1 conv layer before further processes
        x = upsample_wrapper_v1(x, last_conv_feature_maps, upsample_method, kernel=last_conv_kernel)

        # if nb_branch > 1:
        # not separate but use the same.
        # cov_input = SeparateConvolutionFeatures(nb_branch)(x)
        # else:
        #     cov_input = [x]

        # Duplicate Cov for different O2T branches.
        cov_input = nb_branch * [x, ]
        for ind2, xx in enumerate(cov_input):
            cov_branch_y = cov_branch_fn(x, cov_branch_output, stage=ind2, block=str(ind),
                                         **merge_dicts(kwargs, cov_branch_kwargs))
            cov_output.append(cov_branch_y)

        # Merge the sub-branches to vectorize it
        cov_outputs.append(
            merge_branches_with_method(concat=sub_branch_concat,
                                       cov_outputs=cov_output,
                                       cov_output_dim=cov_branch_output,
                                       cov_output_vectorization=cov_output_vectorization,
                                       # Pass the pv parameters into it.
                                       **cov_branch_kwargs
                                       )
        )

    if mode == 3:
        """ basic mutiple cov-branch with one loss, use concat methods """
        if concat == 'concat':
            x = merge.concatenate(cov_outputs)
        else:
            x = merge.add(cov_outputs)
        model_outputs = [Dense(nb_class, activation='softmax')(x)]
    elif mode == 4:
        """ Multiple loss implementation """
        model_outputs = []
        for ind, x in enumerate(cov_outputs):
            x = Dense(nb_class, activation='softmax', name='prediction_{}'.format(ind))(x)
            model_outputs.append(x)

    else:
        raise ValueError("General SO Mode {} not supported".format(mode))

    model = Model(base_model.input, model_outputs, name=name)
    return model



