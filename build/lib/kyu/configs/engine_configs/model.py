"""
Define the model config

"""
from kyu.configs.generic import KCConfig


class ModelConfig(KCConfig):
    """
    Generic Model Config Object containing basic information

    """

    def __init__(self,
                 class_id,
                 model_id,
                 input_shape=(224, 224, 3),
                 nb_outputs=1,       # Add support for multiple losses implementation
                 loss_weights=[1.0,],  # weights passed in
                 name=None):
        self.class_id = class_id
        self.model_id = model_id
        self.input_shape = input_shape
        self.nb_outputs = nb_outputs
        if len(loss_weights) < self.nb_outputs:
            loss_weights = [1.0, ] + (nb_outputs - 1) * [0.2, ]
        self.loss_weights = loss_weights
        self.name = name

    @property
    def target_size(self):
        return self.input_shape[0:2]

    # TODO add classid as property: check the avaliability.
