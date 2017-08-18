'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['KERAS_BACKEND'] = 'theano'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import logging
import sys

from kyu.utils.example_engine import ExampleEngine
from keras.applications.resnet50 import ResNet50CIFAR, ResCovNet50CIFAR
from kyu.models.so_cnn_helper import covariance_block_vector_space
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import Convolution2D, MaxPooling2D, O2Transform
from keras.layers import Dense, Dropout, Activation, Flatten, SecondaryStatistic, WeightedVectorization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.data_utils import get_absolute_dir_project
from kyu.utils.logger import Logger
from kyu.models.cifar import model_original, model_snd, cifar_fitnet_v1, cifar_fitnet_v3, cifar_fitnet_v5, \
    cifar_fitnet_v4

import keras.backend as K

if K._BACKEND == 'tensorflow':
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')

batch_size = 32
nb_classes = 10
nb_epoch = 10
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

BASELINE_PATH = get_absolute_dir_project('model_saved/cifar10_baseline.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_cnn_sndstat.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_fitnet.weights')
LOG_PATH = get_absolute_dir_project('model_saved/log')

cifar_10 = True
label_mode = 'fine'
if cifar_10:
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
else:
    # the data, shuffled and split between train and test sets
    # label_mode = 'fine'
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
    if label_mode is 'fine':
        print('use cifar 100 fine')
        nb_classes = 100
    elif label_mode is 'coarse':
        print('use cifar 100 coarse')
        nb_classes = 20

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

input_shape = X_train.shape[1:]



def resnet50_original():
    # img_input = Input(shape=(img_channels, img_rows, img_cols))
    # model = ResNet50(True, weights='imagenet')
    model = ResNet50CIFAR(nb_class=nb_classes)
    # let's train the model using SGD + momentum (how original).
    return model


def resnet50_snd(parametric=False):
    x, img_input = ResNet50CIFAR(False, nb_class=nb_classes)
    x = SecondaryStatistic(parametrized=False)(x)
    if parametric:
        x = O2Transform(output_dim=100, activation='relu')(x)
        x = O2Transform(output_dim=10, activation='relu')(x)
    x = WeightedVectorization(output_dim=nb_classes, activation='softmax')(x)
    model = Model(img_input, x, name='ResNet50CIFAR_snd')
    return model


def run_original(load=False, save=True, verbose=1):
    model = model_original()
    fit_model(model, load=load, save=save, verbose=verbose)


def run_snd_layer(load=False, save=True, parametric=True):
    model = model_snd(parametric)
    fit_model(model, load=load, save=save)


def run_fitnet_layer(load=False, save=True, second=False, parametric=True, verbose=1):
    model = cifar_fitnet_v1(second=second, parametric=parametric)
    fit_model(model, load=load, save=save, verbose=verbose)


def run_resnet50_original(verbose=1):
    model = resnet50_original()
    fit_model(model, verbose=verbose)


def run_resnet_snd(parametric=True, verbose=1):
    model = resnet50_snd(parametric)
    fit_model(model, load=True, save=True, verbose=verbose)


def run_resnet_merge(parametrics=[], verbose=1, start=0, stop=3):
    for mode in range(start, stop):
        model = ResCovNet50CIFAR(parametrics=parametrics, nb_class=nb_classes, mode=mode)
        fit_model(model, load=False, save=True, verbose=verbose)


def run_fitnet_merge(parametrics=[], verbose=1, start=0, stop=6, cov_mode='o2transform',
                     mode_list=None, epsilon=0, title='cifar', dropout=True, cov_output=None,
                     init='glorot_uniform',
                     cov_mode_input=3,
                     dense_after_covariance=True,
                     cifar_version=3,
                     **kwargs):
    """
    Run Fit-net merge layer testing. All testing cases could be passed throught this interface.
    With all environment settings.


    Parameters
    ----------
    parametrics
    verbose
    start
    stop
    cov_mode
    mode_list
    epsilon
    title
    dropout
    cov_output
    dense_after_covariance

    Returns
    -------

    """
    # TODO Use kwargs to pass the parameters.

    # Deprecate the epsilons

    print('epsilon = ' + str(epsilon))
    if mode_list is None:
        mode_list = range(start=start, stop=stop)
    for mode in mode_list:
        # if mode in [3,5,7]:
        #     print('skip cov_mode {}'.format(mode))
        #     continue
        if cifar_version == 3:
            model = cifar_fitnet_v3(parametrics=parametrics, epsilon=epsilon, mode=mode,
                                    nb_classes=nb_classes, dropout=dropout, init=init,
                                    cov_mode=cov_mode, cov_branch_output=cov_output,
                                    cov_block_mode=cov_mode_input,
                                    dense_after_covariance=dense_after_covariance)
        elif cifar_version == 4:
            model = cifar_fitnet_v4(parametrics=parametrics, epsilon=epsilon, mode=mode,
                                    nb_classes=nb_classes, dropout=dropout, init=init,
                                    cov_mode=cov_mode, cov_branch_output=cov_output,
                                    cov_block_mode=cov_mode_input,
                                    dense_after_covariance=dense_after_covariance, **kwargs)
        elif cifar_version == 5:
            model = cifar_fitnet_v5(parametrics=parametrics, epsilon=epsilon, mode=mode,
                                    nb_classes=nb_classes, dropout=dropout, init=init,
                                    cov_mode=cov_mode, cov_branch_output=cov_output,
                                    dense_after_covariance=dense_after_covariance)
        else:
            print('cifar version not supported ' + str(cifar_version))
            return
        fit_model(model, load=False, save=True, verbose=verbose, title=title, **kwargs)


def fit_model(model, load=False, save=True, verbose=1, title='cifar10', batch_size=32):

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    save_log = True
    engine = ExampleEngine([X_train, Y_train], model, [X_test, Y_test],
                           load_weight=load, save_weight=save, save_log=save_log,
                           lr_decay=True, early_stop=True, tensorboard=True,
                           batch_size=batch_size, nb_epoch=nb_epoch, title=title, verbose=verbose)

    if save_log:
        sys.stdout = engine.stdout
    model.summary()
    engine.fit(batch_size=batch_size, nb_epoch=nb_epoch, augmentation=data_augmentation)
    score = engine.model.evaluate(X_test, Y_test, verbose=0)
    # engine.plot_result('loss')
    engine.plot_result()
    print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))
    if save_log:
        sys.stdout = engine.stdout.close()


def run_routine1():
    print('Test original')
    run_original()
    # print("Test snd model without pre-loading data")
    # run_snd_layer()
    # print('Test snd model with pre-trained')
    # run_snd_layer(loads=True)

def run_routine2():
    """
    Train model snd_layer without pretraining
    :return:
    """
    print("test snd layer")
    sys.stdout = Logger(LOG_PATH + '/cifar_routine2.log')
    run_snd_layer(load=False)


def run_routine3():
    sys.stdout = Logger(LOG_PATH + '/cifar_routine3.log')
    run_snd_layer(load=True, save=False)


def run_routine4():
    # sys.stdout = Logger(LOG_PATH + '/cifar_routine4.log')
    run_resnet50_original(2)


def run_routine5():
    # sys.stdout = Logger(LOG_PATH + '/cifar_routine4.log')
    # run_fitnet_layer(load=True, verbose=1)
    run_fitnet_layer(second=True, load=False, verbose=2)


def run_routine6():
    logging.info("Run holistic test for merged model")
    print("Run holistic test for merged model")
    run_resnet_merge([],2)
    run_resnet_merge([50],2)
    run_resnet_merge([100],2)
    run_resnet_merge([100, 50],2)


def run_routine7():
    logging.info("Run holistic test for complex merged model 2")
    print("Run holistic test for complex merged model 2")
    print("cov_mode 5 for para reset")
    # run_resnet_merge([],2, 5, 6)
    run_resnet_merge([50],2, 5, 6)
    run_resnet_merge([100],2, 5, 6)
    run_resnet_merge([100, 50],2, 5, 6)


def run_routine8():
    """ Fitnet complete trial """
    # paras = [[100,], [50,], [100,50]]
    # paras = [[50,], [100,50]]
    paras = [[100,50]]
    print("Run routine 8 Epoch {} for para {}".format(nb_epoch, paras))
    list_model = []
    for para in paras:
        model = cifar_fitnet_v1(True, para)
        list_model.append(model)
    list_model.append(cifar_fitnet_v1(False, []))
    for model in list_model:
        print('test model {}'.format(model.name))
        fit_model(model, load=False, save=True, verbose=2)


def run_routine9():
    """
    Fitnet v2 complete test
    2016.12.2 Not testing the epsilon yet due to the work priority

    Returns
    -------

    """
    epsilon = 0
    # print("routine 9, epsilon and 6 -  epoch {}".format(nb_epoch))
    print("routine 9, epsilon {} and  epoch {}".format(epsilon, nb_epoch))
    # run_fitnet_merge([], 1, 1, 6, epsilon=1e-4) # Mode 1 for epsilon fails.
    # run_fitnet_merge([],1, 6, 8) # cov_mode 6,7
    # run_fitnet_merge([], 1, 1, 8, epsilon=1e-4)
    run_fitnet_merge([50,], 1, 1, 8, epsilon=epsilon)
    run_fitnet_merge([100,], 1, 1, 8, epsilon=epsilon)
    run_fitnet_merge([50, 100], 1, 0, 8, epsilon=epsilon)


def run_routine10():
    """
    Try the backend as Tensorflow

    Returns
    -------

    """
    nb_epoch = 50
    # model = cifar_fitnet_v1(False)
    model = cifar_fitnet_v3([], mode=0, input_shape=(32, 32, 3))
    fit_model(model, load=False, save=False, verbose=2)


def run_routine11():
    """
    Test the CIFAR fitnet model
    Returns
    -------

    """
    nb_epoch = 200
    from kyu.models.cifar import cifar_fitnet_v2
    model = cifar_fitnet_v2([], mode=8)
    fit_model(model, load=False, save=True, verbose=2)


def run_routine12():
    """
    Create for CIFAR 100 Test
    Parametric test first
    Then non-parametric, multiple model of parametric
    Still have bugs when doing
    Returns
    -------

    """
    nb_epoch = 200
    # param = [16, 8]
    # param = [16, 32]
    # run_fitnet_merge([], 1, mode_list=[2,4,6,8])
    # run_fitnet_merge([], 1, mode_list=[0], title='cifar10-baseline-nodropout')
    # run_fitnet_merge([50], 1, mode_list=[4,2,3,1,5,6,7,8])
    # run_fitnet_merge([50,20], 1, mode_list=[4,2,3,1,5,6,7,8], title='cifar10-cov-dense')
    for param in [[16,32], [16,8]]:
        print("Routine 12 nb_epoch {} paramatric mode {}".format(nb_epoch,param))
        # run_fitnet_merge(param, 1, mode_list=[2], title='cifar10-cov-o2t', cov_mode='o2transform')
        run_fitnet_merge(param, 1, mode_list=[0], title='cifar10-baseline', cov_mode='dense')

    # run_fitnet_merge([100], 1, mode_list=[2])
    # run_fitnet_merge([100], 1, mode_list=[1,2])


def run_routine13():
    """
    Create for CIFAR 10 test
    2016.12.14

    Made for a complete test case for the following steps:
    1.  Model information:
            test Cov-branch: SecondStat - (O2Transform) - WP (cov_output) - Dense(10)
                where cov_output == 500
        Specifications:
            parameters: [ [], [50], [100], [100,50], [16, 8], [32, 16], [16, 32]]
            cov_output = [500, 100, 50]
            title = 'cifar10_cov_o2t_wp_dense'
            init='glorot_normal'
        Experiments:
            2016.12.20  Mode 1 vs Mode 0 (where mode 0 is only run for cov_output - times
            2017.1.3    3 Layer models from mode 1 to 9.

        # Run 2 2016.12.15 -> Should run non-para after this

        # Test 3-layer model

    2.
    Returns
    -------

    """
    # params = [[50], [100], [100,50], [16, 8]]
    params = [[100,100,100], [50,50,50],[25,25,25], [64,32,16],[100,50,25]]   # Exp 2
    # params = [[25,25,25]]                             # Exp 2
    # params = [[100,50], [50,25], [100,75]]
    # params = [[64,32,16],[100,50,25]]                 # Exp 2
    # params = [[], [100], [32, 16], [16, 32]]
    ### Systematic experiments
    # params = [[], [50], [100], [100,50], [16, 8], [32, 16], [16, 32]]   # exp 3
    # params = [[]]
    nb_epoch = 200
    # cov_outputs = [10]
    cov_outputs = [100]
    for cov_output in cov_outputs:
        for param in params:
            print("Run routine 13 nb epoch {} param mode {}".format(nb_epoch, param))
            print('Initialize with glorot_normal')
            # run_fitnet_merge(param, mode_list=[1,2,3,4,5,6,7,8,9],
            run_fitnet_merge(param, mode_list=[1],
                             title='cifar10_cov_o2t_wp{}_dense_nodropout'.format(str(cov_output)),
                             cov_mode='o2transform', dropout=False, init='glorot_normal',
                             cov_output=cov_output)


def run_routine14():
    """
    Create for CIFAR 10 test
    2017.1.10

    Made for interesting findings verifications:
    1.  Model information:
            test Cov-branch: SecondStat - (O2Transform) - WP (cov_output) - Dense(10)
                where cov_output == previous O2Transform number
        Specifications:
            parameters: [ [50], [100], [100,50], [16, 8], [32, 16], [16, 32]]
            cov_output = [ 50, 100, 50, 8, 16, 32 ] for each model
            title = 'cifar10_cov_o2t_wp{}_dense'
            init='glorot_normal'
        Experiments:
            2017.1.10 : Test all param with cov_output correspondingly
            2017.1.10 : Test all params with mode 10, with cov_output=10
            2017.1.10 : Test with mode 11, obtain the baseline.
            2017.1.17 : Test Cov-input from different block.
                            100,50 0.1 percentage improve, test with all settings
            2017.1.17 : Test complete 3 Cov-branch with average wp
    2.
    Returns
    -------

    """
    nb_epoch = 200

    # params = [[100, 50], [50], [100],  [16, 8], [32, 16], [16, 32]] # exp 1,2,5
    # params = [[]] # exp 3
    params = [[100,50]] # exp 6
    # params = [[50], [100],  [16, 8], [32, 16], [16, 32]] # exp 4 [100,50] tested

    # mode_list = [1,2]   # exp 1
    # mode_list = [10]    # exp 2
    # mode_list = [11]    # exp 3
    # mode_list = [2]     # exp 4
    # mode_list = [13]    # exp 5
    mode_list = [2]       # exp 6

    # cov_output = None   # exp 1, 3, 4 (not important)
    # cov_output = 10     # exp 2, 5
    cov_output = 50     # exp 6

    # cov_mode_input = 3  # exp 1,2,3
    # cov_mode_input = 1  # exp 4, 5(not important)
    cov_mode_input = 2  # exp 6 compare 1,2,3 block with 100-50

    for param in params:
        print("Run routine 14 nb epoch {} param mode {} mode list {}".format(nb_epoch, param, mode_list))
        print('Initialize with glorot_normal')
        if cov_output is None and len(param) > 0:
            cov_output = param[-1]
        else:
            cov_output = nb_classes
        run_fitnet_merge(param, mode_list=mode_list,
                         title='cifar10_cov_o2t_wp{}_dense_nodropout_block_{}'.format(
                             str(cov_output), str(cov_mode_input)),
                         cov_mode_input=cov_mode_input,
                         cov_mode='o2transform', dropout=False, init='glorot_normal',
                         cov_output=cov_output)

def run_routine15():
    """
    Create for new ideas:
        Experiment 1: 2016.1.19
            Veryfication of new idea. Separate Convolution blocks for 1st and 2nd order information.
            Right after the input.


    Returns
    -------

    """
    nb_epoch = 200
    exp = 3
    if exp == 1:
        params = [[100, 50], [128, 64], [512,256], [100,100], [64,64], [128,64,32]] # exp 1
        mode_list = [1]
        cov_outputs = [100, 50, 10]
    elif exp == 2:
        params = [[100, 50], [128, 64], [512,256], [100,100], [64,64], [128,64,32]] # exp 1
        mode_list = [1]
        cov_outputs = [100, 50, 10]
    elif exp == 3:
        params = [[], [100], [50]] # exp 1
        mode_list = [1]
        cov_outputs = [100, 50, 10]
    else:
        return
    print("Running experiment {}".format(exp))
    for param in params:
        for mode in mode_list:
            for cov_output in cov_outputs:
                print("Run routine 15 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                run_fitnet_merge(param, mode_list=[mode],
                                 title='cifar10_cov_o2t_wp{}_two_branch'.format(
                                     str(cov_output)
                                 ),
                                 cov_mode_input=3,
                                 cifar_version=5,
                                 cov_mode='o2transform',dropout=False, init='glorot_normal',
                                 cov_output=cov_output)


def run_routine16():
    """
    Create for testing the Residual covariance learning.

    Returns
    -------

    """
    nb_epoch = 200
    exp = 1
    if exp == 1:
        params = [[100, 100, 100]]
        mode_list = [1]
        cov_outputs = [50]
        nb_block = 3
        cifar_version = 4
        cov_mode = 'residual'
        dropout=False
    else:
        return
    print("Running experiments {}".format(exp))
    for param in params:
        for mode in mode_list:
            for cov_output in cov_outputs:
                print("Run residual learning for fitnet")
                run_fitnet_merge(param, mode_list=[mode],
                                 title='cifar10_residual_cov_o2t_wv{}'.format(
                                     str(cov_output)
                                 ),
                                 cov_mode_input=3,
                                 cifar_version=cifar_version,
                                 cov_mode=cov_mode, dropout=dropout, init='glorot_normal',
                                 cov_output=cov_output,
                                 batch_size=128)


def print_model_structures():
    nb_epoch = 200
    exp = 1
    if exp == 1:
        params = [[]]  # exp 1
        mode_list = [1]
        cov_outputs = [50]

    else:
        return
    print("Running experiment {}".format(exp))
    for param in params:
        for mode in mode_list:
            for cov_output in cov_outputs:
                print("Print the routine 15 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                run_fitnet_merge(param, mode_list=[mode],
                                 title='cifar10_cov_o2t_wp{}_two_branch'.format(
                                     str(cov_output)
                                 ),
                                 cov_mode_input=3,
                                 cifar_version=3,
                                 cov_mode='o2transform', dropout=False, init='glorot_normal',
                                 cov_output=cov_output,
                                 batch_size=128)

if __name__ == '__main__':
    nb_epoch = 200
    # run_routine1()
    # print('test')
    # run_routine1()
    # run_routine5()
    # run_routine6()
    # sys.stdout = logging
    # run_routine7()
    # run_resnet50_original(2)
    # run_resnet_snd(True, verbose=1)
    # plot_models()
    # run_routine8()
    # run_routine9()
    # run_routine11()
    # plot_rescov_results()
    # run_routine12()
    run_routine13()
    # run_routine10()
    # run_routine14()
    # run_routine15()
    # print_model_structures()
    # run_routine16()