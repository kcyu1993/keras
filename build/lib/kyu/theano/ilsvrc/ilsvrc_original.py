"""
Training pipeline for ImageNet dataset.
Keras model.

Use backend Tensorflow

"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['KERAS_BACKEND'] = 'theano'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys
from kyu.utils.example_engine import ExampleEngine
from keras.optimizers import SGD
import keras.backend as K

from kyu.datasets.imagenet import ImageNetLoader
from keras.preprocessing.image import ImageDataGeneratorAdvanced, ImageDataGenerator

from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')

# Absolute paths
IMAGENET_PATH = '/home/kyu/.keras/datasets/ILSVRC2015'
TARGET_SIZE = (224,224)
RESCALE_SMALL = 256

BATCH_SIZE = 16
NB_EPOCH = 70
VERBOSE = 1
SAVE_LOG = False

# ImageNet generator
imageNetLoader = ImageNetLoader(IMAGENET_PATH)
gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                 horizontal_flip=True,
                                 channelwise_std_normalization=True)

train = imageNetLoader.generator('train', image_data_generator=gen)
valid = imageNetLoader.generator('valid', image_data_generator=gen)
# test = imageNetLoader.generator('valid', image_data_generator=gen)


def fit_model(model, load=True, save=True, title='imagenet'):
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    engine = ExampleEngine(train, model, valid,
                           load_weight=load, save_weight=save, save_log=SAVE_LOG,
                           lr_decay=True, early_stop=True, tensorboard=True,
                           batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, title=title, verbose=VERBOSE)

    if SAVE_LOG:
        sys.stdout = engine.stdout
    model.summary()
    engine.fit(batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, augmentation=False)
    # score = engine.model.evaluate(X_test, Y_test, verbose=0)

    # engine.plot_result('loss')
    engine.plot_result()
    # print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))
    if SAVE_LOG:
        sys.stdout = engine.stdout.close()


def runroutine1():
    """
    Baseline result. VGG 16
    Returns
    -------

    """

    # model = VGG16()
    model = ResNet50(weights=None)
    fit_model(model)

def test_runtine1():
    model = VGG16()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    data, label = train.next()
    score = model.evaluate(data, label, batch_size=16, verbose=1)

    print(score)



if __name__ == '__main__':
    runroutine1()
    # test_runtine 1()