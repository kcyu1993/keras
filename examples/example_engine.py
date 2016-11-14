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


    """
    def __init__(self, data, model, load_weight=True, save_weight=True, verbose=2,
                 logfile=None):


    def getlogfiledir(self):
