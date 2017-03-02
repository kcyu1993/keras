from configobj import ConfigObj

import os


def create_configObj(filename, dataset, model, runconfigs):
    """
    Create config file given a file name

    Returns
    -------

    """
    config = ConfigObj()
    config.filename = filename

    config['dataset'] = dataset
    config['model'] = model
    config['runconfigs'] = runconfigs
    config.write()


class RunConfig(object):
    """
    Run config object is fed to ExampleEngine v2.
    Consists of two sub-configuration

    """

    def __init__(self, modelConfig, dataConfig,
                 nb_epoch=200,
                 batch_size=32,
                 optimizer='sgd',
                 opt_kwards={},
                 loss='categorical_crossentropy',
                 metrics=['accuracy'],
                 save_log=True,
                 augmentation=False,

                 **kwargs):
        self.modelConfig = modelConfig
        self.dataConfig = dataConfig

        # Run Config passed into ExampleEngine init,
        # apart from model and data
        self.runConfig = kwargs

    def save_to_config(self, filename):
        """
        Save the current object into a filename.config file

        Parameters
        ----------
        filename object saved

        Returns
        -------

        """
        create_configObj(filename, self.modelConfig, self.dataConfig, self.runConfig)

    @classmethod
    def read_from_config(cls, filename):
        """
        Resume a Run-Config from pre-define config file.

        Parameters
        ----------
        filename : str  path to config file read

        Returns
        -------

        """


class DCovConfig(object):
    """
    Config for Secondary DCov experiments.
    return ResNet50_o2(parametrics=param, mode=mode, nb_classes=config.nb_class, cov_branch_output=cov_output,
                       input_shape=config.input_shape, last_avg=config.last_avg, freeze_conv=config.freeze_conv,
                       cov_regularizer=config.cov_regularizer, last_conv_feature_maps=config.last_conv_feature_maps,
                       nb_branch=config.nb_branch, **kwargs
                       )
    """
    def __init__(self,
                 params=[],
                 mode_list=[],
                 cov_outputs=[],
                 cov_branch='o2transform',
                 cov_mode='channel',
                 early_stop=True,
                 cov_regularizer=None,
                 nb_branch=1,
                 dropout=False,
                 last_conv_feature_maps=[],
                 batch_size=8,
                 exp=0,
                 vectorization='wv',
                 epsilon=0,
                 title='',
                 **kwargs
                 ):
        self.__dict__.update(locals())


class ModelConfig(object):
    """
    Model Config Object containing all information to create a model

    """
    def __init__(self, model, mode, comments, keras_config):
        self.model = model
        self.mode = mode
        self.comments = comments
        self.keras_config = keras_config


class DataConfig(object):
    """
    Data Config Object containing all information to get a data object
    """
    pass


if __name__ == '__main__':
    pass