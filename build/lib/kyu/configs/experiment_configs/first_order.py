"""
Experiment Configs Instances
    For First-order baseline models
        * VGG 16

"""
from kyu.configs.model_configs import VggFirstOrderConfig


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
