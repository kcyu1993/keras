import os
from configobj import ConfigObj

from kyu.utils.io_utils import ProjectFile


class KCConfig(ConfigObj):
    """
     define SuperClass Config
     add file system support. Save to the default places ??
    """
    @property
    def default_location(self):
        return self.default_location

    @default_location.setter
    def default_location(self, value):
        if os.path.exists(value):
            self.default_location = value
        else:
            raise ValueError("location not found {}".format(value))

    @property
    def configname(self):
        if self.configname is None:
            return "default.config"
        return self.configname

    def write(self, outfile=None, section=None):
        """ Wrapper with self defined locations if default is not supported. """
        if outfile is None:
            outfile = os.path.join(self.default_location, self.configname)
        super(KCConfig, self).write(outfile, section)


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
        save per epoch ?
        early stop
        lr decay

        tensorboards
    """
    def __init__(self,
                 root_folder,
                 running_title=None,
                 batch_size=32,
                 verbose=2,
                 lr_decay=True,
                 early_stop=True,
                 save_weights=True,
                 load_weights=False,
                 init_weights_location=None,
                 save_per_epoch=True,

                 optimizer='SGD',
                 lr=0.01,
                 dcov_config=None, # possibly
                 ):
        self.__dict__.update(locals())
        self._title = running_title if dcov_config is None else dcov_config.title

    @property
    def title(self):
        return self.title

    @title.setter
    def title(self, value):
        self._title = value if value is not None else self._title
