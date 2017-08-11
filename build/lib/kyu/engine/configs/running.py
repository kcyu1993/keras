import os
from configobj import ConfigObj

from kyu.utils.io_utils import ProjectFile


class KCConfig(object):
    """
     define SuperClass Config
     add file system support. Save to the default places ??
    """

    # def __init__(self, default_location, config_name):
    #     self._default_location = default_location
    #     self._configname = config_name
    #
    # @property
    # def default_location(self):
    #     return self._default_location
    #
    # @default_location.setter
    # def default_location(self, value):
    #     if os.path.exists(value):
    #         self._default_location = value
    #     else:
    #         raise ValueError("location not found {}".format(value))
    #
    # @property
    # def configname(self):
    #     if self._configname is None:
    #         return "default.config"
    #     return self._configname

    def write_to_config(self, outfile, section=None):
        """ Wrapper with self defined locations if default is not supported. """
        if outfile is None:
            raise ValueError("Cannot pass None to save method ")
        if os.path.exists(outfile):
            os.remove(outfile)
        with open(outfile, 'w') as f:
            config = ConfigObj()
            config.merge(self.__dict__)
            config.write(f)

    @classmethod
    def load_config_from_file(cls, infile):
        config = ConfigObj(infile, raise_errors=True)
        return cls(**config.dict())


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
                 _title=None,
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
                 dcov_config=None, # possibly
                 ):
        # self.__dict__.update(locals())
        self._title = _title if dcov_config is None else dcov_config.title
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

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value if value is not None else self._title

