import os

from .generic import KCConfig
from kyu.utils.io_utils import ProjectFile


def createDirectHelper(location, source_path=None):
    if os.path.exists(location):
        if os.path.exists(source_path):
            return ProjectFile(location, source_path=source_path)
        else:
            return ProjectFile(location)
    else:
        raise ValueError("Not found path at {}".format(location))


class RunningConfig(KCConfig):
    """
    Record the configuration during training/finetunning the model

        batch_size
        verbose
        save log

        ProjectFile object to have all stored locations
            for model saving (soft-link)
            for running storing



        init weight path (if necessary)
        save weights
        load weights
        save per epoch
        early stop
        lr decay

        tensorboards
    """
    def __init__(self,
                 # root_folder,
                 _title='default',
                 nb_epoch=100,
                 batch_size=32,
                 verbose=2,
                 lr_decay=True,
                 sequence=8,
                 patience=8,
                 early_stop=True,
                 save_weights=True,
                 load_weights=False,
                 init_weights_location=None,
                 save_per_epoch=True,
                 tensorboard=None,
                 optimizer='SGD',
                 lr=0.01,
                 model_config=None, # possibly
                 rescale_small=320,
                 random_crop=True,
                 horizontal_flip=True,
                 tf_debug=False,
                 tf_debug_filters_name=None,
                 tf_debug_filters_func=None,
                 comments='',
                 ):
        # self.__dict__.update(locals())
        self.model_config = model_config
        self._title = _title
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.lr_decay = lr_decay
        self.lr = lr

        self.sequence = sequence
        self.patience = patience

        self.early_stop = early_stop
        self.load_weights = load_weights
        self.init_weights_location = init_weights_location

        self.save_weights = save_weights
        self.save_per_epoch = save_per_epoch

        self.optimizer = optimizer
        self.tensorboard = tensorboard

        self.rescale_small = rescale_small
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip
        self.comments = comments

        self.tf_debug = tf_debug
        self.tf_debug_filters_func = tf_debug_filters_func
        self.tf_debug_filters_name = tf_debug_filters_name

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value if value is not None else self._title

    # TODO add the optimizer to property for smart update the related kwargs
