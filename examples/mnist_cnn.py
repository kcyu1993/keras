'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf


batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


def mnist_cnn_tf():
    """
    Try to implement the blog using CNN

    https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

    Returns
    -------
    Nothing
    """
    with tf.device("/gpu:0"):
        img = tf.placeholder(tf.float32, shape=(None, img_rows * img_cols))
        labels = tf.placeholder(tf.float32, shape=(None, nb_classes))

        x = Dense(128, activation='relu')(img)
        x = Dense(128, activation='relu')(x)
        preds = Dense(nb_classes, activation='softmax')(x)
        from keras.objectives import categorical_crossentropy
        # loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
        loss = K.mean(categorical_crossentropy(labels, preds))
    return img, labels, preds, loss


def tf_run_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session()
    K.set_session(sess)
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    img, labels, preds, loss = mnist_cnn_tf()

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    with sess.as_default():
        for i in range(nb_epoch):
            batch = mnist_data.train.next_batch(batch_size)
            train_step.run(feed_dict={img:batch[0], labels:batch[1]})

    from keras.metrics import categorical_accuracy as accuracy
    acc_value = accuracy(labels, preds)
    with sess.as_default():
        print(acc_value.eval(feed_dict={img: mnist_data.test.images,
                                        labels: mnist_data.test.labels}))


def mnist_cnn_keras():
    """
    Try to use Tensorflow in keras, failed, cannot direct use model.fit
    to use GPU

    Returns
    -------

    """
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    K.clear_session()

    with tf.device("/gpu:0"):
        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        # model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
        #           verbose=1, validation_data=(X_test, Y_test))
        input_tensor = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        output_tensor = model(input_tensor)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(output_tensor)
    # sess.run()

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    tf.logging.set_verbosity(10)
    tf_run_session()