import tensorflow.contrib.slim as slim
import tensorflow as tf
# import tensorflow.python.ops.nn as nn
from tensorflow.python.ops import nn


def fitnet_slim(input_tensor):
    """
    Define the slim model for Fitnet baseline, testing the performance
    Returns
    -------
    logits
    """
    x = slim.conv2d(input_tensor, 16, [3,3], scope='conv1_1')
    x = slim.conv2d(x, 16, [3,3], scope='conv1_2')
    x = slim.conv2d(x, 16, [3,3], scope='conv1_3')
    x = slim.max_pool2d(x,[2,2], scope='pool1')
    x = slim.dropout(x, keep_prob=0.25, scope='dropout')

    x = slim.repeat(x, 3, slim.conv2d, 32, [3,3], scope='conv2')
    x = slim.max_pool2d(x, [2,2], scope='pool2')

    x = slim.repeat(x, 2, slim.conv2d, 48, [3,3], scope='conv3')
    x = slim.conv2d(x, 64, [3,3], scope='conv3_3')
    x = slim.flatten(x)
    x = slim.fully_connected(x, 500, scope='fc1')
    x = slim.fully_connected(x, 10, activation_fn=nn.softmax, scope='prediction')
    return x


def losses(a,b):
    losses = slim.losses

