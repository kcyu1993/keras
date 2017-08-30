"""
Define visualization methods here

"""

import tensorflow as tf

from keras.applications import ResNet50

# LOG_PATH_ROOT = '/home/kyu/cvkyu/tensorboard/'
from kyu.legacy.resnet50 import ResNet50_o2_multibranch, ResCovNet50

LOG_PATH_ROOT = '/Users/kcyu/tensorboard/'


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

    model = ResNet50_o2_multibranch([256, 128, 64], mode=3, nb_classes=23, cov_mode='pmean', cov_branch='o2t_no_wv',
                                    cov_branch_output=64, concat='matrix_diag')
    # model = ResNet50_o2_multibranch([256, 128, 64], mode=2, nb_classes=23, cov_mode='pmean', cov_branch='o2transform',
    #                                 cov_branch_output=64, concat='concat')

    # model = ResCovNet50()
    outputs = model.outputs

    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.summary()
    # log_model(model, path='test/resnet')