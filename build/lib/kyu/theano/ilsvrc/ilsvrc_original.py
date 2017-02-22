"""
Training pipeline for ImageNet dataset.
Keras model.

Use backend Tensorflow

"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['CNMEM'] = '1'
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import sys
from keras.optimizers import SGD
import keras.backend as K

from kyu.datasets.imagenet import ImageNetLoader, preprocess_image_for_imagenet
from keras.preprocessing.image import ImageDataGeneratorAdvanced, ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50

from kyu.utils.example_engine import ExampleEngine
from kyu.models.ilsvrc1000_vgg import VGG16_with_second
from kyu.models.resnet import ResCovNet50

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')

# Absolute paths
IMAGENET_PATH = '/home/kyu/.keras/datasets/ILSVRC2015'
TARGET_SIZE = (224,224)
RESCALE_SMALL = 256

BATCH_SIZE = 4
NB_EPOCH = 1000
VERBOSE = 1
SAVE_LOG = True
VALIDATION = True
# ImageNet generator
imageNetLoader = ImageNetLoader(IMAGENET_PATH)
gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                 horizontal_flip=True,
                                 preprocessing_function=preprocess_image_for_imagenet
                                 # channelwise_std_normalization=True
                                 )


train = imageNetLoader.generator('train', image_data_generator=gen, batch_size=BATCH_SIZE)
valid = imageNetLoader.generator('valid', image_data_generator=gen, batch_size=BATCH_SIZE)
# test = imageNetLoader.generator('valid', image_data_generator=gen)


def fit_model(model, load=True, save=True, title='imagenet', nb_epoch=NB_EPOCH):
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    engine = ExampleEngine(train, model, valid,
                           load_weight=load, save_weight=save, save_log=SAVE_LOG,
                           lr_decay=True, early_stop=True, tensorboard=True,
                           batch_size=BATCH_SIZE, nb_epoch=nb_epoch, title=title, verbose=VERBOSE)

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
    print("Fit VGG16 Base-line test with Images")
    model = VGG16()
    # model = ResNet50(weights=None)
    fit_model(model)


def runroutine2():
    """
    Test VGG16 with Second layer structure.
    Non-parametric case
    Returns
    -------

    """
    # mode = 2 # Exp 3, try to use secondary information earlier
    mode = 1 # Exp 2, try naive version of Cov net

    # print("Fit mode 1, simple replacement with last VGG layer. ") # Exp 1
    print("Fit mode {}, simple replacement with last VGG layer. ".format(mode)) # Exp 2 trainfrom scratch
    model = VGG16_with_second(mode=mode, cov_mode='o2transform', weights=None)
    fit_model(model)


def run_routine3():
    """
    Test ResNet with cov branch

        Experiments:
            2017.1.17 : Test Baseline ResNet with Cov-version mode 1
            2017.1.18 : Test With Mode 1 again.
            2017.1.19 : Test Independent learning branch for covariance descriptor

    Returns
    -------

    """
    nb_epoch = 200
    exp = 3
    indep = False
    if exp == 1 or 3:
        mode = 1                 # Exp 1 (mode = 1,0)
        params = [[1000,1000], [2048,1000], [4096,1000]]     # Exp 1,2
        cov_input = 3
        if exp == 3:
            indep = True
    elif exp == 2:
        mode = 2                # Exp 2
        cov_input = 2           # Exp 2
        params = [[128,64]]     # Exp 1,2
    else :
        return

    print("==================================================")
    print("ImageNet test with residual net exp {} mode {} ".format(exp, mode))
    print("==================================================")
    for param in params:
        model = ResCovNet50(nb_classes=1000, mode=mode, parametrics=param, cov_block_mode=cov_input,
                            independent_learning=indep)
        fit_model(model, nb_epoch=nb_epoch, title='imagenet'.format(cov_input))


def test_runtine1():
    model = VGG16()
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    data, label = train.next()
    score = model.evaluate(data, label, batch_size=16, verbose=1)

    print(score)



if __name__ == '__main__':
    # runroutine1()
    # test_runtine 1()
    # runroutine2()
    run_routine3()