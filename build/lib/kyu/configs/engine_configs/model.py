"""
Define the model config

"""
from .generic import KCConfig


class ModelConfig(KCConfig):
    """
    Generic Model Config Object containing basic information

    """

    def __init__(self, class_id, model_id, input_shape=(224, 224, 3), name=None):
        self.class_id = class_id
        self.model_id = model_id
        self.input_shape = input_shape
        self.name = name

    @property
    def target_size(self):
        return self.input_shape[0:2]

    # TODO add classid as property: check the avaliability.