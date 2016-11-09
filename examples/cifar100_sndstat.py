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
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SecondaryStatistic, WeightedProbability
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.data_utils import get_absolute_dir_project
from keras.utils.logger import Logger
import sys

batch_size = 32
nb_classes = 20
nb_epoch = 100
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

BASELINE_PATH = get_absolute_dir_project('model_saved/cifar10_baseline.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_cnn_sndstat.weights')
LOG_PATH = get_absolute_dir_project('model_saved/log')

# the data, shuffled and split between train and test sets
# label_mode = 'fine'
label_mode = 'coarse'
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
if label_mode is 'fine':
    nb_classes = 100
elif label_mode is 'coarse':
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

def model_original():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def model_snd():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(SecondaryStatistic(activation='linear'))
    model.add(WeightedProbability(10,activation='linear', init='normal'))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def fit(model):
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                                         batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test))
        return model

def test_original():
    model = model_original()
    model = fit(model)
    model.summary()
    model.save_weights(BASELINE_PATH)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))

def test_snd_layer(load=False, save=True):
    model = model_snd()
    if load:
        model.load_weights(BASELINE_PATH, by_name=True)
    else:
        model = fit(model)
    model.summary()
    if save:    model.save_weights(SND_PATH)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))

def test_merge_model():
    raise NotImplementedError


def test_routine1():
    print('Test original')
    test_original()
    # print("Test snd model without pre-loading data")
    # test_snd_layer()
    # print('Test snd model with pre-trained')
    # test_snd_layer(loads=True)

def test_routine2():
    """
    Train model snd_layer without pretraining
    :return:
    """
    print("test snd layer")
    sys.stdout = Logger(LOG_PATH + '/cifar_routine2.log')
    test_snd_layer(load=False)

def test_routine3():
    sys.stdout = Logger(LOG_PATH + '/cifar_routine3.log')
    test_snd_layer(load=True, save=False)

if __name__ == '__main__':
    # test_routine1()
    # print('test')
    test_routine1()
