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
from keras.preprocessing.image import ImageDataGenerator, Iterator

from kyu.utils.io_utils import ProjectFile


class ClassificationTrainer(object):
    """
    Add the function step by step.

        1. Basic training function (without test)
            Support model compilation

        2. Log the model and related information to assigned folder
        3.
    """
    def __init__(self, model, train,
                 validation=None, test=None,
                 model_config=None,
                 data_config=None,
                 running_config=None,
                 logger=None,
                 root_folder=None,
                 ):
        self.model = model # Ready for compilation
        self.train = train
        self.validation = validation
        self.test = test
        self.dirhelper = ProjectFile(root_folder)

        self.model_config = model_config
        self.running_config = running_config
        self.logger = logger
        self.data_config = data_config

        self.mode = self.get_data_mode() # mode for fit ndarray or generator

    def build(self):
        """ Construct the corresponding configs to prepare running """
        # Keras Callbacks
        pass

    def fit(self, batch_size=32, nb_epoch=100, verbose=2, augmentation=True):


    def get_data_mode(self):
        # Data input
        data = self.train
        if isinstance(data, (list, tuple)):
            assert len(data) == 2
            self.train = data
        elif isinstance(data, (ImageDataGenerator, Iterator)):
            self.train = data
            self.mode = 1
        else:
            raise TypeError('data is not supported {}'.format(data.__class__))
        return self.mode

    # Get folder names methods
    def get_plot_folder(self):
        return self.dirhelper.get_plot_path(self.train.name)

    def get_weight_folder(self):
        return self.dirhelper.get_run_path(self.train.name, self.model.name, self.model_config.cov_branch)

