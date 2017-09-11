"""
Similar to ModelConfig. Implement a simple data config for later extension. 

"""
from .generic import KCConfig


class DatasetConfig(KCConfig):
    def __init__(self, 
                 name,
                 dirpath=None,
                 **kwargs):
        self.__dict__.update(locals())

