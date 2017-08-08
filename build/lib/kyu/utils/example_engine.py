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

import logging

import keras.backend as K

from keras.callbacks import ModelCheckpoint, \
    CSVLogger, LearningRateScheduler, \
    EarlyStopping, ReduceLROnPlateau, TensorBoard

from kyu.utils.data_utils import get_absolute_dir_project, get_weight_path
from kyu.utils.io_utils import cpickle_load, cpickle_save
from kyu.utils.logger import Logger
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator, Iterator

import os
import sys

from kyu.utils.callback import ReduceLROnDemand


def getlogfiledir():
    return get_absolute_dir_project('model_saved/log')


def gethistoryfiledir():
    return get_absolute_dir_project('model_saved/history')

def hist_average(historys):
    raise NotImplementedError


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
            cross validation for selected parameters


    """
    def __init__(self, data, model, validation=None, test=None,
                 load_weight=True, save_weight=True,
                 lr_decay=False, early_stop=False,
                 save_per_epoch=False, tensorboard=False,
                 batch_size=128, nb_epoch=100,
                 verbose=2, logfile=None, save_log=True,
                 weight_path='',
                 title='default'):
        """

        Parameters
        ----------
        data    (X, Y) or Generator
        model   Keras.Model
        validation  same as data
        test        same as data
        load_weight True to support loading weights
        save_weight True to support saving weights
        save_per_epoch  True to save the weight after each epochs
        batch_size      # Should not be called.
        nb_epoch        # should not be called
        verbose         verbose same as keras, 0 quiet, 1 detail, 2 brief
        logfile         LogFile location
        save_log        True to save log
        title           Title of all saving files.
        """

        self.model = model
        self.title = title
        self.mode = 0 # 0 for fit ndarray, 1 for fit generator

        # Data input
        if isinstance(data, (list,tuple)):
            assert len(data) == 2
            self.train = data
        elif isinstance(data, (ImageDataGenerator, Iterator)):
            self.train = data
            self.mode = 1
        else:
            raise TypeError('data is not supported {}'.format(data.__class__))
            return

        # Load the validation and test data.
        if self.mode == 0:
            if validation is not None:
                assert len(validation) == 2
            if test is not None:
                assert len(test) == 2

        if self.mode == 1:
            if validation is not None:
                assert isinstance(validation, (Iterator,))
                self.nb_te_sample = validation.nb_sample
            else:
                self.nb_te_sample = 0
            if test is not None:
                assert isinstance(test, (ImageDataGenerator, Iterator))

        self.model_label = ['com', 'gen']

        self.validation = validation
        self.test = test

        self.verbose = verbose
        self.logfile = logfile
        self.log_flag = save_log
        self.default_stdout = sys.stdout
        if logfile is None:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}.log".format(
                self.title, model.name, self.mode))
        else:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}_{}.log".format(
                self.title, model.name, self.mode, logfile))

        if self.log_flag:
            self.stdout = Logger(self.logfile)
            self.verbose = 2

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        # Set the weights
        if weight_path == '' or weight_path is None or weight_path =='imagenet':
            self.weight_path = get_weight_path(
                "{}-{}_{}.weights".format(self.title, model.name, self.mode),
                'dataset')
        else:
            self.weight_path = weight_path
        logging.debug("weights path {}".format(self.weight_path))
        self.save_weight = save_weight
        self.load_weight = load_weight
        self.save_per_epoch = save_per_epoch
        if not os.path.exists(self.weight_path):
            print("weight not found, create a new one or transfer from current weight")
            self.load_weight = False
        else:
            if self.load_weight == True:
                print("Weight path found at {}".format(self.weight_path))
        # Keras Callback
        self.cbks = []
        if self.save_weight and self.save_per_epoch:
            self.cbks.append(ModelCheckpoint(self.weight_path + ".tmp", verbose=1))

        self.lr_decay = lr_decay
        if self.lr_decay:
            print("Reduced LR on DEMAND ")
            self.cbks.append(ReduceLROnDemand(min_lr=1e-6, factor=0.5, sequence=8,
                                              verbose=2, epsilon=1e-5))
        else:
            print("Reduced LR on PLATEAU")
            self.cbks.append(ReduceLROnPlateau(min_lr=1e-6, verbose=1, patience=8))
        self.early_stop = early_stop
        if self.early_stop:
            self.cbks.append(EarlyStopping(patience=20, verbose=1))

        self.tensorboard = tensorboard
        if self.tensorboard:
            if K._BACKEND == 'tensorflow':
                tb_path = '/tmp/tensorflow/' + title[:30]
                print("Creating tensorboard to save to {}".format(tb_path))
                self.cbks.append(TensorBoard(log_dir=tb_path, histogram_freq=1, write_images=False))

    def fit(self, batch_size=32, nb_epoch=100, verbose=2, augmentation=True):

        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        if nb_epoch == 0:
            return None

        if self.load_weight:
            print("Load weights from {}".format(self.weight_path))
            self.model.load_weights(self.weight_path, by_name=True)
        try:
            # Handle ModelCheckPoint
            if self.mode == 0:
                history = self.fit_ndarray(augmentation)
            elif self.mode == 1:
                history = self.fit_generator()
            else:
                history = None
        except (KeyboardInterrupt, SystemExit):
            print("System catch do some thing (like save the model)")
            if self.save_weight:
                print("Weights saved to {}".format(self.weight_path + '.tmp'))
                self.model.save_weights(self.weight_path + ".tmp", overwrite=True)
                try:
                    history = self.model.history
                    self.save_history(history, tmp=True)
                    self.plot_result(tmp=True)
                except Exception:
                    raise Exception
            raise

        self.history = history
        if self.save_weight and self.save_per_epoch:
            # Potentially remove the tmp file.
            os.remove(self.weight_path + '.tmp')
            print("Weights saved to {}".format(self.weight_path))
            self.model.save_weights(self.weight_path, overwrite=True)
        if self.log_flag:
            sys.stdout.close()

        return history

    def fit_generator(self):
        print("{} fit with generator".format(self.model.name))
        sample_per_epoch = self.train.n
        if sample_per_epoch == 0:
            sample_per_epoch = 128 * 200
        history = self.model.fit_generator(
            self.train, samples_per_epoch=sample_per_epoch, nb_epoch=self.nb_epoch,
            nb_worker=4,
            validation_data=self.validation, nb_val_samples=self.nb_te_sample,
            verbose=self.verbose,
            callbacks=self.cbks
        )
        if self.save_weight:
            print("weights saved to {}".format(self.weight_path))
            self.model.save_weights(self.weight_path)
            self.model.save_weights(self.weight_path + "_" + str(self.nb_epoch))
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
        else:
            valid = None
        if not augmentation:
            print('Not using data augmentation.')
            # shape = (nb_samples, channel, img_w, img_h)
            # (nb_samples,) + vectorized_categories  # 1 -> [0,1,0], 2
            hist = self.model.fit(X_train, Y_train,
                                  batch_size=self.batch_size,
                                  nb_epoch=self.nb_epoch,
                                  validation_data=valid,
                                  shuffle=True,
                                  callbacks=self.cbks)
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
            if valid:
                datagen_valid = ImageDataGenerator(
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
                datagen_valid.fit(valid[0])

                # fit the model on the batches generated by datagen.flow()
                hist = self.model.fit_generator(datagen.flow(X_train, Y_train,
                                                             batch_size=self.batch_size),
                                                samples_per_epoch=X_train.shape[0],
                                                nb_epoch=self.nb_epoch,
                                                verbose=self.verbose,
                                                validation_data=datagen_valid.flow(valid[0], valid[1],
                                                                                   batch_size=self.batch_size),
                                                nb_val_samples=len(valid[0]),
                                                callbacks=self.cbks)
            else:
                hist = self.model.fit_generator(datagen.flow(X_train, Y_train,
                                                             batch_size=self.batch_size),
                                                samples_per_epoch=X_train.shape[0],
                                                nb_epoch=self.nb_epoch,
                                                verbose=self.verbose,
                                                callbacks=self.cbks)
            if self.save_weight:
                self.model.save_weights(self.weight_path)
        return hist

    def plot_result(self, metric=('loss', 'acc'), linestyle='-', show=False, dictionary=None, tmp=False):
        """
        Plot result according to metrics passed in
        Parameters
        ----------
        metric : list or str  ('loss','acc') or one of them
        linestyle
        show
        dictionary

        Returns
        -------

        """
        if tmp:
            import numpy as np
            filename = "{}-{}_{}_{}.png".format(self.title, self.model.name, self.mode, np.random.randint(1,1e4))
        else:
            filename = "{}-{}_{}.png".format(self.title, self.model.name, self.mode)
        if self.history is None:
            return
        history = self.history.history
        if dictionary is not None:
            history = dictionary
        if isinstance(metric, str):
            if metric is 'acc':
                train = history['acc']
                valid = history['val_acc']
            elif metric is 'loss':
                train = history['loss']
                valid = history['val_loss']
            else:
                raise RuntimeError("plot only support metric as loss, acc")
            x_factor = range(len(train))
            from keras.utils.visualize_util import plot_train_test
            from _tkinter import TclError
            try:
                plot_train_test(train, valid, x_factor=x_factor, show=show,
                                xlabel='epoch', ylabel=metric,
                                linestyle=linestyle,
                                filename=filename, plot_type=0)
                self.save_history(history)
            except TclError:
                print("Catch the Tcl Error, save the history accordingly")
                self.save_history(history)
                return
        else:
            assert len(metric) == 2
            tr_loss = history['loss']
            tr_acc = history['acc']
            va_loss = history['val_loss']
            va_acc = history['val_acc']
            x_factor = range(len(tr_loss))
            from keras.utils.visualize_util import plot_loss_acc
            from _tkinter import TclError
            try:
                plot_loss_acc(tr_loss, va_loss, tr_acc=tr_acc, te_acc=va_acc, show=show,
                              xlabel='epoch', ylabel=metric,
                              filename=filename)
                self.save_history(history)
            except TclError:
                print("Catch the Tcl Error, save the history accordingly")
                self.save_history(history)
                return

    def save_history(self, history, tmp=False):
        from keras.callbacks import History
        import numpy as np

        if isinstance(history, History):
            history = history.history
        filename = "{}-{}_{}.history".format(self.title, self.model.name, np.random.randint(1e4))

        if tmp:
            filename = 'tmp_' + filename
        if history is None:
            return
        logging.debug("compress with gz")
        dir = gethistoryfiledir()
        if not os.path.exists(dir):
            os.mkdir(dir)
        filename = os.path.join(dir, filename)
        print("Save the history to " + filename)
        filename = cpickle_save(data=history, output_file=filename)
        return filename

    @staticmethod
    def load_history(filename):
        logging.debug("load history from {}".format(filename))
        hist = cpickle_load(filename)
        if isinstance(hist, dict):
            return hist
        else:
            raise ValueError("Should read a dict term")

    @staticmethod
    def load_history_from_log(filename):
        logging.debug('Load history from {}'.format(filename))
        with open(filename, 'r') as f:
            lines = [line.rstrip() for line in f]
            hist = dict()
            hist['loss'] = []
            hist['val_loss'] = []
            hist['acc'] = []
            hist['val_acc'] = []
            for line in lines:
                if not line.startswith('Epoch'):
                    for i, patch in enumerate(line.replace(' ', '').split('-')[1:]):
                        n, val = patch.split(":")
                        hist[n].append(float(val))
        return hist


class CVEngine(ExampleEngine):

    def __init__(self, models, data, validation=None, test=None,
                 load_weight=True, save_weight=True,
                 save_per_epoch=False,
                 verbose=2, logfile=None, save_log=True,
                 title='cvalid_default'):
        super(CVEngine, self).__init__(data, None, validation=validation, test=test,
                                       load_weight=load_weight, save_weight=save_weight,
                                       save_per_epoch=save_per_epoch, verbose=verbose, logfile=logfile,
                                       save_log=save_log, title=title)
        # Override the parameters
        self.__dict__.update(locals())

        self.mode = 0  # 0 for fit nd-array, 1 for fit generator
        if isinstance(data, (list, tuple)):
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

        # Override part
        if logfile is not None:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}.log".format(
                self.title, 'crossvalid', self.mode))
        else:
            self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}_{}.log".format(
                self.title, "crossvalid", self.mode, logfile))

        # Set the weights
        self.weight_path = get_weight_path(
            "{}-{}_{}.cv.weights".format(self.title, "cross_valid", self.mode),
            'dataset')
        logging.debug("weights path {}".format(self.weight_path))

        if not os.path.exists(self.weight_path):
            print("weight not found, create a new one or transfer from current weight")
            self.load_weight = False

        self.models = models
        self.model_label = ['com', 'gen']

    def cross_validation(self, models=[], k_fold=1, nb_epoch=50, batch_size=16):
        """

        Parameters
        ----------
        models:  [keras.Model]
                    if models is [] or None, the cross_validation won't work
        k_fold:      k_fold validation, default = 1
        nb_epoch:
        batch_size:

        Returns
        -------
            result: dict    dictionary as format {'model1.name': history}
        """
        if models is None or []:
            models = self.models
        result = dict()
        for ind, model in enumerate(models):
            hist_list = []
            for k in range(k_fold):
                raise NotImplementedError




# class ExampleEnginev2(object):
#     # TODO ASAP
#
#     """
#     Goal:
#         Unified test purpose for secondary image statistics.
#         Separate the model creation and testing purpose
#         Reduce code duplication.
#
#     Update:
#         Support config file generation
#         Support ProjectFile function to simplify directory organization
#         Support resuming from breaking.
#
#
#     Function:
#         Give a
#             model as well as its ConfigObj
#             data as well as its ConfigObj
#         Implement:
#             Keras training pipeline
#             fit function for data array / generator
#             evaluate function
#             weights path and load/save
#             related saving operations, like weights, history.
#                 TO be specific, epochs should be saved,
#                 To support resuming training like tensorflow.
#
#     Requirements:
#         1. Conform to FileOrganization structure
#         2. Generate RunConfig, ModelConfig and DataConfig
#         3. Save
#
#
#     """
#     def __init__(self, data, model, validation=None, test=None,
#                  model_config=None, data_config=None,
#                  load_weight=True, save_weight=True,
#                  lr_decay=True, early_stop=False,
#                  save_per_epoch=False, tensorboard=False,
#                  batch_size=128, nb_epoch=100,
#                  verbose=2, logfile=None, save_log=True,
#                  title='default'):
#         """
#
#         Parameters
#         ----------
#         data            (X, Y) or Generator
#         model           Keras.Model
#         validation      same as data
#         test            same as data
#         model_config    ConfigObj
#         data_config     ConfigObj
#         load_weight     True to support loading weights
#         save_weight     True to support saving weights
#         save_per_epoch  True to save the weight after each epochs
#         batch_size      # Should not be called.
#         nb_epoch        # should not be called
#         verbose         verbose same as keras, 0 quiet, 1 detail, 2 brief
#         logfile         LogFile location
#         save_log        True to save log
#         title           Title of all saving files.
#         """
#
#         self.model = model
#         self.title = title
#         self.mode = 0 # 0 for fit ndarray, 1 for fit generator
#
#         # Data input
#         if isinstance(data, (list,tuple)):
#             assert len(data) == 2
#             self.train = data
#         elif isinstance(data, (ImageDataGenerator, Iterator)):
#             self.train = data
#             self.mode = 1
#         else:
#             raise TypeError('data is not supported {}'.format(data.__class__))
#             return
#
#         # Load the validation and test data.
#         if self.mode == 0:
#             if validation is not None:
#                 assert len(validation) == 2
#             if test is not None:
#                 assert len(test) == 2
#
#         if self.mode == 1:
#             if validation is not None:
#                 assert isinstance(validation, (Iterator,))
#                 self.nb_te_sample = validation.nb_sample
#             else:
#                 self.nb_te_sample = 0
#             if test is not None:
#                 assert isinstance(test, (ImageDataGenerator, Iterator))
#
#         self.model_label = ['com', 'gen']
#
#         self.validation = validation
#         self.test = test
#
#         self.verbose = verbose
#         self.logfile = logfile
#         self.log_flag = save_log
#         self.default_stdout = sys.stdout
#         if logfile is None:
#             self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}.log".format(
#                 self.title, model.name, self.mode))
#         else:
#             self.logfile = os.path.join(getlogfiledir(), "{}-{}_{}_{}.log".format(
#                 self.title, model.name, self.mode, logfile))
#
#         if self.log_flag:
#             self.stdout = Logger(self.logfile)
#             self.verbose = 2
#
#         self.nb_epoch = nb_epoch
#         self.batch_size = batch_size
#
#         # Set the weights
#         self.weight_path = get_weight_path(
#             "{}-{}_{}.weights".format(self.title, model.name, self.mode),
#             'dataset')
#         logging.debug("weights path {}".format(self.weight_path))
#         self.save_weight = save_weight
#         self.load_weight = load_weight
#         self.save_per_epoch = save_per_epoch
#         if not os.path.exists(self.weight_path):
#             print("weight not found, create a new one or transfer from current weight")
#             self.load_weight = False
#
#         # Keras Callback
#         self.cbks = []
#         if self.save_weight and self.save_per_epoch:
#             self.cbks.append(ModelCheckpoint(self.weight_path + ".tmp", verbose=1))
#
#         self.lr_decay = lr_decay
#         if self.lr_decay:
#             # self.cbks.append(ReduceLROnPlateau(min_lr=0.0001, verbose=1))
#             self.cbks.append(ReduceLROnDemand(min_lr=1e-6, verbose=1, sequence=8, factor=0.25))
#
#         self.early_stop = early_stop
#         if self.early_stop:
#             self.cbks.append(EarlyStopping(patience=20, verbose=1))
#
#         self.tensorboard = tensorboard
#         if self.tensorboard:
#             if K._BACKEND == 'tensorflow':
#                 tb_path = '/tmp/tensorflow/' + title
#                 print("Creating tensorboard to save to {}".format(tb_path))
#                 self.cbks.append(TensorBoard(log_dir=tb_path, histogram_freq=1, write_images=False))
#
#     def fit(self, batch_size=32, nb_epoch=100, verbose=2, augmentation=False):
#
#         self.batch_size = batch_size
#         self.nb_epoch = nb_epoch
#         if self.load_weight:
#             print("Load weights from {}".format(self.weight_path))
#             self.model.load_weights(self.weight_path, by_name=True)
#         try:
#             # Handle ModelCheckPoint
#             if self.mode == 0:
#                 history = self.fit_ndarray(augmentation)
#             elif self.mode == 1:
#                 history = self.fit_generator()
#             else:
#                 history = None
#         except (KeyboardInterrupt, SystemExit):
#             print("System catch do some thing (like save the model)")
#             if self.save_weight:
#                 print("Weights saved to {}".format(self.weight_path + '.tmp'))
#                 self.model.save_weights(self.weight_path + ".tmp", overwrite=True)
#                 try:
#                     history = self.model.history
#                     self.save_history(history, tmp=True)
#                 except Exception:
#                     raise Exception
#             raise
#
#         self.history = history
#         if self.save_weight and self.save_per_epoch:
#             # Potentially remove the tmp file.
#             os.remove(self.weight_path + '.tmp')
#
#         if self.log_flag:
#             sys.stdout.close()
#
#         return history
#
#     def fit_generator(self):
#         print("{} fit with generator".format(self.model.name))
#         history = self.model.fit_generator(
#             self.train, samples_per_epoch=128*200, nb_epoch=self.nb_epoch,
#             nb_worker=4,
#             validation_data=self.validation, nb_val_samples=self.nb_te_sample,
#             verbose=self.verbose,
#             callbacks=self.cbks)
#         if self.save_weight:
#             print("weights saved to {}".format(self.weight_path))
#             self.model.save_weights(self.weight_path)
#             self.model.save_weights(self.weight_path + "_" + str(self.nb_epoch))
#         self.history = history
#         return self.history
#
#     def fit_ndarray(self, augmentation=True):
#         print('model fitting with ndarray data')
#         X_train = self.train[0]
#         Y_train = self.train[1]
#         if self.test is not None:
#             X_test = self.test[0]
#             Y_test = self.test[1]
#
#         if self.validation is not None:
#             valid = self.validation
#         if not augmentation:
#             print('Not using data augmentation.')
#             # shape = (nb_samples, channel, img_w, img_h)
#             # (nb_samples,) + vectorized_categories  # 1 -> [0,1,0], 2
#             hist = self.model.fit(X_train, Y_train,
#                                   batch_size=self.batch_size,
#                                   nb_epoch=self.nb_epoch,
#                                   validation_data=valid,
#                                   shuffle=True,
#                                   callbacks=self.cbks)
#         else:
#             print('Using real-time data augmentation.')
#             # this will do preprocessing and realtime data augmentation
#             datagen = ImageDataGenerator(
#                 featurewise_center=False,  # set input mean to 0 over the dataset
#                 samplewise_center=False,  # set each sample mean to 0
#                 featurewise_std_normalization=False,  # divide inputs by std of the dataset
#                 samplewise_std_normalization=False,  # divide each input by its std
#                 zca_whitening=False,  # apply ZCA whitening
#                 rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#                 width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#                 height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#                 horizontal_flip=True,  # randomly flip images
#                 vertical_flip=False)  # randomly flip images
#
#             # compute quantities required for featurewise normalization
#             # (std, mean, and principal components if ZCA whitening is applied)
#             datagen.fit(X_train)
#
#             # fit the model on the batches generated by datagen.flow()
#             hist = self.model.fit_generator(datagen.flow(X_train, Y_train,
#                                                          batch_size=self.batch_size),
#                                             samples_per_epoch=X_train.shape[0],
#                                             nb_epoch=self.nb_epoch,
#                                             verbose=self.verbose,
#                                             validation_data=valid,
#                                             callbacks=self.cbks)
#             if self.save_weight:
#                 self.model.save_weights(self.weight_path)
#         return hist
#
#     def plot_result(self, metric=('loss', 'acc'), linestyle='-', show=False, dictionary=None):
#         """
#         Plot result according to metrics passed in
#         Parameters
#         ----------
#         metric : list or str  ('loss','acc') or one of them
#         linestyle
#         show
#         dictionary
#
#         Returns
#         -------
#
#         """
#         filename = "{}-{}_{}.png".format(self.title, self.model.name, self.mode)
#         if self.history is None:
#             return
#         history = self.history.history
#         if dictionary is not None:
#             history = dictionary
#         if isinstance(metric, str):
#             if metric is 'acc':
#                 train = history['acc']
#                 valid = history['val_acc']
#             elif metric is 'loss':
#                 train = history['loss']
#                 valid = history['val_loss']
#             else:
#                 raise RuntimeError("plot only support metric as loss, acc")
#             x_factor = range(len(train))
#             from keras.utils.visualize_util import plot_train_test
#             from _tkinter import TclError
#             try:
#                 plot_train_test(train, valid, x_factor=x_factor, show=show,
#                                 xlabel='epoch', ylabel=metric,
#                                 linestyle=linestyle,
#                                 filename=filename, plot_type=0)
#                 self.save_history(history)
#             except TclError:
#                 print("Catch the Tcl Error, save the history accordingly")
#                 self.save_history(history)
#                 return
#         else:
#             assert len(metric) == 2
#             tr_loss = history['loss']
#             tr_acc = history['acc']
#             va_loss = history['val_loss']
#             va_acc = history['val_acc']
#             x_factor = range(len(tr_loss))
#             from keras.utils.visualize_util import plot_loss_acc
#             from _tkinter import TclError
#             try:
#                 plot_loss_acc(tr_loss, va_loss, tr_acc=tr_acc, te_acc=va_acc, show=show,
#                               xlabel='epoch', ylabel=metric,
#                               filename=filename)
#                 self.save_history(history)
#             except TclError:
#                 print("Catch the Tcl Error, save the history accordingly")
#                 self.save_history(history)
#                 return
#
#     def save_history(self, history, tmp=False):
#         from keras.callbacks import History
#         import numpy as np
#
#         if isinstance(history, History):
#             history = history.history
#         filename = "{}-{}_{}.history".format(self.title, self.model.name, np.random.randint(1e4))
#
#         if tmp:
#             filename = 'tmp_' + filename
#         if history is None:
#             return
#         logging.debug("compress with gz")
#         dir = gethistoryfiledir()
#         if not os.path.exists(dir):
#             os.mkdir(dir)
#         filename = os.path.join(dir, filename)
#         print("Save the history to " + filename)
#         filename = cpickle_save(data=history, output_file=filename)
#         return filename
#
#     @staticmethod
#     def load_history(filename):
#         logging.debug("load history from {}".format(filename))
#         hist = cpickle_load(filename)
#         if isinstance(hist, dict):
#             return hist
#         else:
#             raise ValueError("Should read a dict term")
#
#     @staticmethod
#     def load_history_from_log(filename):
#         logging.debug('Load history from {}'.format(filename))
#         with open(filename, 'r') as f:
#             lines = [line.rstrip() for line in f]
#             hist = dict()
#             hist['loss'] = []
#             hist['val_loss'] = []
#             hist['acc'] = []
#             hist['val_acc'] = []
#             for line in lines:
#                 if not line.startswith('Epoch'):
#                     for i, patch in enumerate(line.replace(' ', '').split('-')[1:]):
#                         n, val = patch.split(":")
#                         hist[n].append(float(val))
#         return hist
