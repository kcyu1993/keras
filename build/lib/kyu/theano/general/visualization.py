"""
Define visualization methods here

"""

import tensorflow as tf

from keras.applications import ResNet50
from keras.applications.resnet50 import ResCovNet50

LOG_PATH_ROOT = '/home/kyu/cvkyu/tensorboard/'

def log_model(model, path='.log'):
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=True
    ))
    with sess.as_default():
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        print("print to path " + LOG_PATH_ROOT + path)
        writer = tf.summary.FileWriter(LOG_PATH_ROOT + path, graph=sess.graph)
        writer.flush()
    return writer


if __name__ == '__main__':

    model = ResCovNet50()
    outputs = model.outputs
    model.compile(optimizer='sgd', loss='categorical_crossentropy')

    log_model(model, path='test/resnet')