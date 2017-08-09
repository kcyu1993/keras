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
                 ):
        pass
