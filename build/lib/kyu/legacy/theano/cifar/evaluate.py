import time

from kyu.theano.cifar.configs import get_experiment_settings

import keras.backend as K
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.optimizers import SGD
from keras.utils import np_utils
from kyu.models.fitnet import fitnet_o1, fitnet_o2
from keras.utils.data_utils import get_absolute_dir_project

nb_classes = 10

if K.backend() == 'tensorflow':
    input_shape=(32,32,3)
    K.set_image_dim_ordering('tf')
else:
    input_shape=(3,32,32)
    K.set_image_dim_ordering('th')

TARGET_SIZE = (32,32)


batch_size = 32
nb_classes = 10
nb_epoch = 10
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

input_shape = (32, 32, 3)
BASELINE_PATH = get_absolute_dir_project('model_saved/cifar10_baseline.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_cnn_sndstat.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_fitnet.weights')
LOG_PATH = get_absolute_dir_project('model_saved/log')

def cifar10_data():

    cifar_10 = True
    label_mode = 'fine'
    if cifar_10:
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        nb_classes = 10
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

    # input_shape = X_train.shape[1:]
    return [X_train, Y_train], [X_test, Y_test]


def cifar_evaluate(model,
                   weights_path=None,
                   title='cifar-evaluate',
                   optimizer=None,
                   ):

    train, test = cifar10_data()

    if optimizer is None:
        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path, True)
    stime = time.time()
    result = model.evaluate(test[0], test[1], batch_size=batch_size)
    print("--- %s seconds ---" % (time.time() - stime))
    print(title)
    print(result)


def evaluate_cifar_baseline():
    model = fitnet_o1(1, [500], input_shape=input_shape)
    cifar_evaluate(model, title='baseline cifar fitnet')


def evaluate_cifar_second():

    config = get_experiment_settings(5)
    model = fitnet_o2(config.params[0], mode=1, input_shape=input_shape, cov_mode=config.cov_mode,
                      cov_branch_output=config.cov_outputs[0], nb_classes=nb_classes)
    cifar_evaluate(model, title='so-cnn-4')


if __name__ == '__main__':
    # evaluate_cifar_baseline()
    evaluate_cifar_second()