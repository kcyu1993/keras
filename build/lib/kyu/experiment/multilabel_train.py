"""
Train a multilabel network

Specifically made for ChestXray training pipeline
"""


from keras.layers import Conv2D, BatchNormalization
from keras.optimizers import SGD
from kyu.engine.trainer import ClassificationTrainer
from kyu.engine.utils.callbacks import TensorBoardWrapper
from kyu.models import get_model
from kyu.models.so_cnn_helper import get_tensorboard_layer_name_keys
from kyu.utils.image import get_vgg_image_gen, get_resnet_image_gen, get_densenet_image_gen
from kyu.utils.io_utils import ProjectFile



def multilabel_train():
    """
    Wrapper for multilabel train.
    Returns
    -------

    """
