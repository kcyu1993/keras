"""
Experiment Configs Instances
    For First-order baseline models
        * VGG 16

"""
from kyu.configs.model_configs import VggFirstOrderConfig, DenseNetFirstOrderConfig


def get_fo_vgg_exp(exp=1):
    if exp == 1:
        return VggFirstOrderConfig(
            nb_classes=0,
            input_shape=(224,224,3),
            denses=[],
            input_tensor=None,
            weights='imagenet',
            include_top=True,
            freeze_conv=False,
            last_pooling=True,
            name='VGG-16-baseline'
        )
    elif exp == 2:
        return VggFirstOrderConfig(
            nb_classes=0,
            input_shape=(224, 224, 3),
            denses=[4096, 4096],
            # denses=[1024, 1024],
            # denses=[256, 256],
            # denses=[512, 512],
            # denses=[],
            input_tensor=None,
            weights='imagenet',
            include_top=False,
            freeze_conv=False,
            last_pooling=True,
            name='VGG-16-notop-baseline'
        )


def get_fo_dense_exp(exp=1):
    if exp == 1:
        return DenseNetFirstOrderConfig(
            nb_class=0,
            input_shape=(224, 224, 3),
            nb_dense_block=4,
            growth_rate=32,
            nb_filter=64,
            reduction=0.5,
            dropout_rate=0.0,
            weight_decay=1e-4,
            weights_path='imagenet',
            freeze_conv=True,
            last_pooling=True,
            name='DenseNet121-FO'
        )
    else:
        raise ValueError("Not supported exp number {}".format(exp))

