"""
Experiment Configs Instances
    For First-order baseline models
        * VGG 16

"""
from kyu.configs.model_configs import VggFirstOrderConfig, DenseNetFirstOrderConfig, ResNetFirstOrderConfig
from kyu.configs.model_configs.first_order import AlexNetFirstOrderConfig


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


def get_fo_resnet_exp(exp=1):
    if exp == 1:
        return ResNetFirstOrderConfig(
            nb_class=0,
            denses=[],
            include_top=False,
            input_shape=(224, 224, 3),
            weights='imagenet',
            input_tensor=None,
            pooling=None,
            last_avg=True,
            weight_decay=0,
            freeze_conv=True,
            class_id='resnet50',
            name='ResNet50-FO'
        )
    elif exp == 2:
        return ResNetFirstOrderConfig(
            nb_class=0,
            denses=[],
            include_top=False,
            input_shape=(224, 224, 3),
            weights='imagenet',
            input_tensor=None,
            pooling=None,
            last_avg=True,
            weight_decay=0,
            freeze_conv=True,
            pred_activation='sigmoid',
            class_id='resnet50',
            name='ResNet50-FO'
        )
    else:
        raise ValueError("FO-ResNet: Not supported exp number {}".format(exp))


def get_fo_alexnet_exp(exp=1):
    if exp == 1:
        return AlexNetFirstOrderConfig(
            nb_classes=0,
            input_shape=(227, 227, 3),
            input_tensor=None,
            weights=None,
            include_top=True,
            freeze_conv=False,
            last_pooling=True,
            name='AlexNet-FO'
        )
    else:
        raise ValueError("FO-Alexnet: Not supported exp number {}".format(exp))
