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
import sys

import os

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.engine import Model
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import Iterator
from kyu.configs.engine_configs.generic import KCConfig
from kyu.configs.engine_configs import ModelConfig
from kyu.configs.engine_configs import RunningConfig
from kyu.engine.utils.data_utils import ImageData
from kyu.utils.callback import ReduceLROnDemand
from kyu.utils.io_utils import ProjectFile, cpickle_load, cpickle_save
from kyu.utils.logger import Logger


def load_history(filename):
    print ("load history from {}".format(filename))
    hist = cpickle_load(filename)
    if isinstance(hist, dict):
        return hist
    else:
        raise ValueError("Should read a dict term")


def load_history_from_log(filename):
    print('Load history from {}'.format(filename))
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
                 save_log=True,
                 logfile=None,

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

        if isinstance(model_config, ModelConfig) or model_config is None:
            self.model_config = model_config
        else:
            raise ValueError("Model config must be a KCConfig got {}".format(model_config))

        if isinstance(running_config, RunningConfig):
            self.running_config = running_config
        else:
            raise ValueError('running config must be RunningConfig type')

        if logfile is None:
            logfile = dirhelper.get_log_path()
        self.logfile = logfile
        self.save_log = save_log
        if self.save_log:
            self.stdout = Logger(self.logfile)

        # TF debug function
        self.tf_debug = running_config.tf_debug
        self.tf_debug_filters_func = running_config.tf_debug_filters_func
        self.tf_debug_filters_name = running_config.tf_debug_filters_name

        self.cbks = []
        self.fit_mode = 0
        # self.mode = self.get_data_mode() # mode for fit ndarray or generator
        self._built = False
        self.history = None

    def build(self):
        """ Construct the corresponding configs to prepare running """

        # Analyze the running config
        # Keras Callbacks

        # Logger

        # Weight load and over-ride pre-load weights again
        if self.running_config.load_weights and os.path.exists(self.running_config.init_weights_location):
            print("Weights load from {}".format(self.running_config.init_weights_location))
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
                self.cbks.append(EarlyStopping(patience=self.running_config.patience, verbose=1))

        # Tensorboard
        if isinstance(self.running_config.tensorboard, TensorBoard):
            print("Add Tensorboard to the callbacks")
            self.cbks.append(self.running_config.tensorboard)

        # Compile the model (even again)
        self.model.compile(optimizer=self.running_config.optimizer, loss=categorical_crossentropy,
                           metrics=['accuracy', top_k_categorical_accuracy])

        self._built = True

    def fit(self, batch_size=None, nb_epoch=None, verbose=None):
        if self._built is not True:
            self.build()

        if nb_epoch is None:
            nb_epoch = self.running_config.nb_epoch
        if verbose is None:
            verbose = self.running_config.verbose
        if batch_size is None:
            batch_size = self.running_config.batch_size

        if self.save_log:
            sys.stdout = self.stdout

        try:
            if self.fit_mode == 0:
                # Save 3 configs
                self.model_config.write_to_config(self.dirhelper.get_config_path('model'))
                self.save_keras_model_config(self.dirhelper.get_config_path('keras'))
                self.running_config.write_to_config(self.dirhelper.get_config_path('run'))

                if self.tf_debug:
                    from tensorflow.python import debug as tf_debug
                    sess = K.get_session()
                    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                    if self.tf_debug_filters_name:
                        for name, func in zip(self.tf_debug_filters_name, self.tf_debug_filters_func):
                            sess.add_tensor_filter(name, func)
                    K.set_session(sess)
                # Run the train function.
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
                    self.save_history(history, tmp=True)
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

        if self.save_log:
            # Close logging after weight saved.
            sys.stdout = self.stdout.close()

        return history

    def _fit_generator(self, batch_size=None, nb_epoch=None, verbose=None):

        train = self.data.get_train(batch_size=batch_size, target_size=self.model_config.target_size)
        if self.data.use_validation:
            valid = self.data.get_valid(batch_size=batch_size, target_size=self.model_config.target_size)
        else:
            valid = self.data.get_test(batch_size=batch_size, target_size=self.model_config.target_size)

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
            epochs=nb_epoch,
            workers=4,
            validation_data=valid,
            validation_steps=val_steps_per_epoch,
            verbose=verbose,
            callbacks=self.cbks
        )

        return history

    def compile_model(self, optimizer=None, lr=None,  **kwargs):
        """ Provide a way to compile the model """
        if optimizer is None:
            optimizer = self.running_config.optimizer
        if lr is None:
            lr = self.running_config.lr
        self.model.compile(optimizer=optimizer, lr=lr, **kwargs)

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
        prefix = 'loss_acc'
        run_id = self.dirhelper.run_id
        filename = prefix + '-' + run_id
        if tmp:
            filename += ".tmp"
        filename += '.png'

        fpath = os.path.join(self.dirhelper.get_plot_folder(), filename)

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
                                filename=fpath, plot_type=0)
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
                              filename=fpath)
                self.save_history(history)
            except TclError:
                print("Catch the Tcl Error, save the history accordingly")
                self.save_history(history)
                return

    def plot_model(self, show_shapes=True, show_layer_names=True):
        from keras.utils.visualize_util import plot, model_to_dot
        from IPython.display import SVG
        filename = os.path.join(self.dirhelper.get_model_run_path, 'model.png')
        plot(self.model, to_file=filename, show_shapes=show_shapes, show_layer_names=show_layer_names)
        dot_data = model_to_dot(self.model, show_shapes=show_shapes, show_layer_names=show_layer_names)
        svg_data = SVG(dot_data.create(prog='dot', format='svg'))
        dot_data.write()

    def save_history(self, history, tmp=False):
        from keras.callbacks import History

        if isinstance(history, History):
            history = history.history
        filename = self.dirhelper.get_history_path()
        if tmp:
            filename = filename + '.tmp'
        if history is None:
            return
        # logging.debug("compress with gz")

        print("Save the history to " + filename)
        filename = cpickle_save(data=history, output_file=filename)
        return filename

    def save_keras_model_config(self, path, model=None, **kwargs):
        """ Save the keras model config to the running folder """
        import json
        if model is None:
            model = self.model

        json_config = model.to_json(**kwargs)
        with open(path, 'w') as f:
            json.dump(json_config, f)
