from configobj import ConfigObj


class KCConfig(ConfigObj):
    """
     define SuperClass Config
     add file system support. Save to the default places ??
    """
    @property
    def default_location(self):
        return None



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