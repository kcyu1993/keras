"""
Keras Trainer based on Example Engine

    Goal:
        Unified test purpose for secondary image statistics.
        Separate the model creation and testing purpose
        Reduce code duplication.

    Update:
        Support config file generation
        Support ProjectFile function to simplify directory organization
        Support resuming from breaking.


    Function:
        Give a
            model as well as its ConfigObj
            data as well as its ConfigObj
        Implement:
            Keras training pipeline
            fit function for data array / generator
            evaluate function
            weights path and load/save
            related saving operations, like weights, history.
                TO be specific, epochs should be saved,
                To support resuming training like tensorflow.

    Requirements:
        1. Conform to FileOrganization structure
        2. Generate RunConfig, ModelConfig and DataConfig
        3. Save

"""
import os

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator, Iterator
from kyu.engine.configs.running import RunningConfig, KCConfig
from kyu.engine.utils.data_utils import ImageData
from kyu.utils.callback import ReduceLROnDemand

from kyu.utils.io_utils import ProjectFile



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


class ClassificationTrainer(object):
    """
    Add the function step by step.

        1. Basic training function (without test)
            Support model compilation

        2. Log the model and related information to assigned folder
        3. Do not support the nd-array loader, only support ClassificationData
            format with generator.

    """
    def __init__(self, model,
                 data,
                 dirhelper,
                 model_config=None,
                 running_config=None,
                 # logger=None,
                 ):
        """

        Parameters
        ----------
        model : keras.Model
        data : ClassificationData
        model_config : DCov_config ??

        running_config : KCConfig
        logger : Logger
        root_folder
        """
        if isinstance(model, Model):
            self.model = model # Ready for compilation
        else:
            raise ValueError("Model must be a Keras.model got {}".format(model))

        if isinstance(data, ImageData):
            self.data = data
        else:
            raise ValueError("Data must be a ImageData got {}".format(data))

        if isinstance(dirhelper, ProjectFile):
            self.dirhelper = dirhelper
        else:
            raise ValueError("dirhelper must be a ProjectFile object ")

        if isinstance(model_config, KCConfig):
            self.model_config = model_config
        else:
            raise ValueError("Model config must be a KCConfig got {}".format(model_config))

        if isinstance(running_config, RunningConfig):
            self.running_config = running_config
        else:
            raise ValueError('running config must be RunningConfig type')

        # self.logger = logger
        self.cbks = []
        self.fit_mode = 0
        # self.mode = self.get_data_mode() # mode for fit ndarray or generator
        self._built = False

    def build(self):
        """ Construct the corresponding configs to prepare running """

        # Analyze the running config
        # Keras Callbacks

        # Logger

        # Weight load and over-ride pre-load weights again
        if self.running_config.load_weights and os.path.exists(self.running_config.init_weights_location):
            self.model.load_weights(self.running_config.init_weights_location, by_name=True)

        # Save weights and call backs
        if self.running_config.save_weights and self.running_config.save_per_epoch:
            self.cbks.append(ModelCheckpoint(self.dirhelper.get_weight_path()))

        # Add learning rate scheduler
        if isinstance(self.running_config.lr_decay, bool):
            if self.running_config.lr_decay:
                print("Reduce LR on DEMAND")
                self.cbks.append(ReduceLROnDemand(min_lr=1e-6, verbose=1, sequence=self.running_config.sequence))
            else:
                print("Reduce LR on PLATEAU")
                self.cbks.append(ReduceLROnPlateau(min_lr=1e-6, verbose=1, patience=self.running_config.patience))
        elif isinstance(self.running_config.lr_decay, (ReduceLROnDemand, ReduceLROnPlateau)):
            self.cbks.append(self.running_config.lr_decay)

        if isinstance(self.running_config.early_stop, EarlyStopping):
            self.cbks.append(self.running_config.early_stop)
        else:
            if self.running_config.early_stop:
                self.cbks.append(EarlyStopping(self.running_config.patience, verbose=1))

        # Tensorboard
        if isinstance(self.running_config.tensorboard, TensorBoard):
            print("Add Tensorboard to the callbacks")
            self.cbks.append(self.running_config.tensorboard)

        self._built = True

    def fit(self, batch_size=32, nb_epoch=100, verbose=2):
        if self._built is not True:
            self.build()
        try:
            if self.fit_mode == 0:
                history = self._fit_generator(batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)
            else:
                raise NotImplementedError("Fit mode other than 0 is not handled {}".format(self.fit_mode))

        except (KeyboardInterrupt, SystemExit) as e1:
            print("Catch the Keyboard Interrupt. ")
            if self.running_config.save_weights:
                print("Temp weights save to {}".format(self.dirhelper.get_tmp_weight_path()))
                self.model.save_weights(self.dirhelper.get_tmp_weight_path())

                try:
                    history = self.model.history
                    save_history(history, tmp=True)
                    self.plot_result(tmp=True)
                except Exception as e:
                    raise e
            raise e1

        self.history = history

        if self.running_config.save_weights:
            weights_path = self.dirhelper.get_weight_path()
            tmp_weight = self.dirhelper.get_tmp_weight_path()
            print("Weights save to {}".format(weights_path))
            if os.path.exists(tmp_weight):
                os.remove(tmp_weight)
            self.model.save_weights(weights_path)

        # TODO handle the logging
        return history

    def _fit_generator(self, batch_size=32, nb_epoch=100, verbose=2):
        train = self.data.get_train()
        valid = self.data.get_valid()

        if not isinstance(train, Iterator):
            raise ValueError("Only support generator for training data")

        steps_per_epoch = train.n / train.batch_size
        val_steps_per_epoch = valid.n / valid.batch_size if valid is not None else 0

        print("{} fit with generator with steps per epoch training {} val {}".
              format(self.model.name, steps_per_epoch, val_steps_per_epoch))

        if steps_per_epoch == 0:
            steps_per_epoch = 200
        history = self.model.fit_generator(
            train,
            # samples_per_epoch=sample_per_epoch,
            steps_per_epoch=steps_per_epoch,
            # nb_epoch=self.nb_epoch,
            epochs=self.running_config.nb_epoch,
            workers=4,
            validation_data=valid,
            validation_steps=val_steps_per_epoch,
            verbose=self.running_config.verbose,
            callbacks=self.cbks
        )

        return history

    # def get_data_mode(self):
    #     # Data input
    #     data = self.train
    #     if isinstance(data, (list, tuple)):
    #         assert len(data) == 2
    #         self.train = data
    #     elif isinstance(data, (ImageDataGenerator, Iterator)):
    #         self.train = data
    #         self.mode = 1
    #     else:
    #         raise TypeError('data is not supported {}'.format(data.__class__))
    #     return self.mode


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
            filename = "{}-{}_{}_{}.png".format(self.title, self.model.name, self.mode, np.random.randint(1, 1e4))
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
            from kyu.utils.visualize_util import plot_train_test
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
            from kyu.utils.visualize_util import plot_loss_acc
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
