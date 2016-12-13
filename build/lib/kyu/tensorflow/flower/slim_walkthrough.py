import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

# import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
import time

# Main slim library
slim = tf.contrib.slim
losses = tf.contrib.losses


def my_cnn(images, num_classes, is_training):  # is_training is not used...
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(images, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 192)
        net = slim.fully_connected(net, num_classes, activation_fn=None)
        return net


def func():

    with tf.Graph().as_default():
        # The model can handle any input size because the first layer is convolutional.
        # The size of the model is determined when image_node is first passed into the my_cnn function.
        # Once the variables are initialized, the size of all the weight matrices is fixed.
        # Because of the fully connected layers, this means that all subsequent images must have the same
        # input size as the first image.
        batch_size, height, width, channels = 3, 28, 28, 3
        images = tf.random_uniform([batch_size, height, width, channels], maxval=1)

        # Create the model.
        num_classes = 10
        logits = my_cnn(images, num_classes, is_training=True)
        probabilities = tf.nn.softmax(logits)

        # Initialize all the variables (including parameters) randomly.
        init_op = tf.initialize_all_variables()

        with tf.Session() as sess:
            # Run the init_op, evaluate the model outputs and print the results:
            sess.run(init_op)
            probabilities = sess.run(probabilities)

    print('Probabilities Shape:')
    print(probabilities.shape)  # batch_size x num_classes

    print('\nProbabilities:')
    print(probabilities)

    print('\nSumming across all classes (Should equal 1):')
    print(np.sum(probabilities, 1))  # Each row sums to 1

