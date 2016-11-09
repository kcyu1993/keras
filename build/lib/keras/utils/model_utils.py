from __future__ import absolute_import

"""
This python file is utilities for loading pre-trained model.
It would provide the path accordingly
"""
import os

def get_model_dir():
    """
    Get current model directory
    :return:
    """
    # current_dir = get_project_dir()
    # return os.path.join(current_dir, 'data')
    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    return os.path.join(datadir_base, 'model')


def convnet_alexnet():
    dir = get_model_dir()
    path = os.path.join(dir, 'convnets-alexnet_weights.h5')
    return path


