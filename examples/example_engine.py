"""
Engine for example standard carring:

Supply with
    model
    input of following type:
        whole dataset as numpy array
        generator
    verbose
    log_file location
    model I/O

"""
from __future__ import absolute_import
from keras.utils.data_utils import get_absolute_dir_project, get_weight_path
from keras.utils.logger import Logger
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

import os
import sys

def getlogfiledir():
    return get_absolute_dir_project('model_saved/log')


class ExampleEngine(object):
    """
    Goal:
        Unified test purpose for secondary image statistics.
        Separate the model creation and testing purpose
        Reduce code duplication.

    Function:
        Give a
            model
            output path for log
            data / data generator
        Implement:
            fit function for data array / generator
            evaluate function
            weights path and load/save


    """
    def __init__(self, data, model, validation=None, test=None,
                 load_weight=True, save_weight=True,
                 batch_size=128, nb_epoch=100,
                 verbose=2, logfile=None, save_log=True,
                 title='default'):

        self.model = model
        self.title = title
        self.mode = 0 # 0 for fit ndarray, 1 for fit generator
        if isinstance(data, (list,tuple)):
            assert len(data) == 2
            self.train = data
        elif isinstance(data, (ImageDataGenerator, DirectoryIterator)):
            self.train = data
            self.mode = 1

        # Load the validation and test data.
        if self.mode == 0:
            if validation is not None:
                assert len(validation) == 2
            if test is not None:
                assert len(test) == 2

        if self.mode == 1:
            if validation is not None:
                assert isinstance(validation, (DirectoryIterator))
                self.nb_te_sample = validation.nb_sample
            if test is not None:
                assert isinstance(test, (ImageDataGenerator, DirectoryIterator))

        self.model_label = ['com', 'gen']

        self.validation = validation
        self.test = test

        self.verbose = verbose
        self.logfile = logfile
        self.log_flag = save_log

        if logfile is not None:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}.log".format(
                self.title, model.name, self.mode))
        else:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}_{}.log".format(
                self.title, model.name, self.mode, logfile))

        if self.log_flag:
            sys.stdout = Logger(self.logfile)
            self.verbose = 2

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        # Set the weights
        self.weight_path = get_weight_path(
            "{}-{}_{}.weights".format(self.title, model.name, self.mode),
            'dataset')
        print("weights path {}".format(self.weight_path))
        self.save_weight = save_weight
        self.load_weight = load_weight
        if not os.path.exists(self.weight_path):
            self.load_weight = False

    def fit(self, batch_size=32, nb_epoch=100, verbose=2, augmentation=False):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        if self.load_weight:
            self.model.load_weights(self.weight_path, by_name=True)
        if self.mode == 0:
            history = self.fit_ndarray(augmentation)
        elif self.mode == 1:
            history = self.fit_generator()
        else:
            history = None
        self.history = history
        return history

    def fit_generator(self):
        print("{} fit with generator".format(self.model.name))
        history = self.model.fit_generator(
            self.train, samples_per_epoch=128*200, nb_epoch=self.nb_epoch,
            nb_worker=4,
            validation_data=self.validation, nb_val_samples=self.nb_te_sample,
            verbose=self.verbose)
        if self.save_weight:
            self.model.save_weights(self.weight_path)
        self.history = history
        return self.history

    def fit_ndarray(self, augmentation=True):
        print('model fitting with ndarray data')
        X_train = self.train[0]
        Y_train = self.train[1]
        if self.test is not None:
            X_test = self.test[0]
            Y_test = self.test[1]

        if self.validation is not None:
            valid = self.validation
        if not augmentation:
            print('Not using data augmentation.')
            hist = self.model.fit(X_train, Y_train,
                                  batch_size=self.batch_size,
                                  nb_epoch=self.nb_epoch,
                                  validation_data=valid,
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
            hist = self.model.fit_generator(datagen.flow(X_train, Y_train,
                                                         batch_size=self.batch_size),
                                            samples_per_epoch=X_train.shape[0],
                                            nb_epoch=self.nb_epoch,
                                            verbose=self.verbose,
                                            validation_data=valid)

            if self.save_weight:
                self.model.save_weights(self.weight_path)
        return hist

    def plot_result(self, metric='acc', show=False):
        if self.history is None:
            return
        history = self.history
        if metric is 'acsc':
            train = history.history['acc']
            valid = history.history['val_acc']
        elif metric is 'loss':
            train = history.history['loss']
            valid = history.history['val_loss']
        else:
            raise RuntimeError("plot only support metric as loss, acc")
        x_factor = range(len(train))
        filename = "{}-{}_{}.png".format(self.title, self.model.name, self.mode)
        from keras.utils.visualize_util import plot_train_test
        plot_train_test(train, valid, x_factor=x_factor, show=show,
                        xlabel='epoch', ylabel=metric,
                        filename=filename, plot_type=0)
