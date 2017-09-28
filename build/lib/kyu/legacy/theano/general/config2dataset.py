"""
Generate Data object from config file

Generalized Data object generator, based on previous implementation.
Give a dataset name and related keywords, like shuffle, xxx
"""

from configobj import ConfigObj
from keras.utils import np_utils


class Data(object):
    """
    Generalized from MincLoader
    Used for any type of image based data (classification purpose)
    """
    def __init__(self, train, valid, test, shuffle, batch_size, name, comments, **kwargs):
        """
        Here, dataset is DataConfig

        Parameters
        ----------
        dataset
        kwargs
        """
        self.train = train
        self.valid = valid
        self.test = test
        self.shuffle = shuffle
        self.batch_size = batch_size

        # Config to get data
        self.name = name
        self.kwargs = kwargs
        self.comments = comments

    def to_config(self):
        """
        Generate DataConfig
        Returns
        -------

        """
        pass


def config2data(config):
    # Get data from its name and kwargs
    if config['name'] == 'cifar10':
        return cifar10(config)
    pass


def cifar10(config, **kwargs):
    from keras.datasets.cifar10 import load_data
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = load_data()
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    train = (X_train, Y_train)
    test = (X_test, Y_test)

    data = Data(train=train, valid=None, test=test, shuffle=False, batch_size=32,
                )

    return data

