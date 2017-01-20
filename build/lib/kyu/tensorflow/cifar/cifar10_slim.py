import tensorflow.contrib.slim as slim
import tensorflow as tf
# import tensorflow.python.ops.nn as nn
from tensorflow.python.ops import nn

# Second layer
from keras.layers.secondstat import SecondaryStatistic, WeightedVectorization, O2Transform


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
    x = slim.dropout(x, keep_prob=0.75, scope='dropout')

    x = slim.repeat(x, 3, slim.conv2d, 32, [3,3], scope='conv2')
    x = slim.max_pool2d(x, [2,2], scope='pool2')

    x = slim.repeat(x, 2, slim.conv2d, 48, [3,3], scope='conv3')
    x = slim.conv2d(x, 64, [3,3], scope='conv3_3')
    x = slim.flatten(x)
    x = slim.fully_connected(x, 500, scope='fc1')
    x = slim.fully_connected(x, 10, activation_fn=nn.softmax, scope='prediction')
    return x


def simple_slim_model(input_tensor):
    x = slim.conv2d(input_tensor, 32, [3,3], scope='conv1',
                    biases_initializer=tf.constant_initializer(0),
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.05)
                    )
    x = slim.conv2d(x, 32, [32, 32], scope='conv2',
                    biases_initializer=tf.constant_initializer(0),
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.05)
                    )
    x = slim.max_pool2d(x, [2,2], scope='pool1')
    x = slim.dropout(x, keep_prob=0.75, scope='dropout')

    x = slim.conv2d(x, 64, [3,3], scope='conv3',
                    biases_initializer=tf.constant_initializer(0),
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.05)
                    )
    x = slim.conv2d(x, 64, [3,3], scope='conv4',
                    biases_initializer=tf.constant_initializer(0),
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.05)
                    )
    x = slim.max_pool2d(x, [2,2], scope='pool2')
    x = slim.dropout(x, keep_prob=0.75, scope='dropout2')
    x = slim.flatten(x)
    x = slim.fully_connected(x, 512, scope='fc1')
    x = slim.fully_connected(x, 10, scope='predictions')
    # x = slim.fully_connected(x, 10, activation_fn=nn.softmax, scope='predictions')
    return x


def simple_second_model(input_tensor):
    x = slim.conv2d(input_tensor, 32, [3,3], scope='conv1',
                    biases_initializer=tf.constant_initializer(0),
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.05)
                    )

    x = slim.conv2d(x, 32, [32, 32], scope='conv2')
    x = slim.max_pool2d(x, [2,2], scope='pool1')
    x = slim.dropout(x, keep_prob=0.75, scope='dropout')

    x = slim.conv2d(x, 64, [3,3], scope='conv3')
    x = slim.conv2d(x, 64, [3,3], scope='conv4')
    x = slim.max_pool2d(x, [2,2], scope='pool2')
    x = slim.dropout(x, keep_prob=0.75, scope='dropout2')
    x = SecondaryStatistic(activation='relu', dim_ordering='tf')(x)
    # x = O2Transform(100, activation='relu')(x)
    # x = O2Transform(50, activation='relu')(x)
    x = WeightedVectorization(100, activation='relu')(x)
    x = slim.fully_connected(x, 10, scope='predictions')
    # x = slim.fully_connected(x, 10, activation_fn=nn.softmax, scope='predictions')
    return x


def retest_lenet(input_tensor):
    x = slim.conv2d(input_tensor, 64, [5,5], scope='conv1')
    x = slim.max_pool2d(x, [3,3], scope='pool1')
    x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    x = slim.conv2d(x, 64, [5,5], scope='conv2')
    x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    x = slim.max_pool2d(x, [3,3], scope='pool2')
    x = slim.flatten(x)
    x = slim.fully_connected(x, 384, scope='fc1')
    x = slim.fully_connected(x, 192, scope='fc2')
    x = slim.fully_connected(x, 10, scope='softmax_linear')
    return x

def losses(a,b):
    losses = slim.losses

