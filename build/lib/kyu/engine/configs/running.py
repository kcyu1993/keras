from configobj import ConfigObj


class KCConfig(ConfigObj):
    """
     define SuperClass Config
     add file system support. Save to the default places ??
    """



class RunningConfig(KCConfig):
    """
    Record the configuration during training/finetunning the model

        batch_size
        verbose
        save log
        logging location
        model saved location
        plot location

        init weight path (if necessary)
        save weights
        load weights
        save per epoch ?
        early stop
        lr decay

        tensorboards
    """