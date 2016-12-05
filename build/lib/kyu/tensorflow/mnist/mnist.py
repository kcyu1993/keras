from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import math
from tensorflow.examples.tutorials.mnist import mnist, input_data

import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# def placeholder_inputs():
#     """
#     Return the input place holder
#     Returns
#     -------
#
#     """
#     images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
#                                                            mnist.IMAGE_PIXELS))
#     labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
#     return images_placeholder, labels_placeholder


def inference(images, hidden1_units, hidden2_units):

    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases'
                             )
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases'
                             )
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels=labels, name='xentropy'
    )
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # Global step is like the actually steps happened ?
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))




