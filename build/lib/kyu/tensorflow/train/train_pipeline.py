"""
Training pipeline build for any Tensorflow Image-Classification tasks

Author: Kaicheng Yu

Methods support:
    Image Generator (either from self-made (MINC) or TF-Record (imagenet)


API (temptive):
    ------ Data and func to be fit ------
    inference : function(TF tensor) -> TF tensor
    loss : function (TF-tensor) -> TF-scalar
    dataset : generator with next(dataset) -> TF-tensor [None, input_shape]

    ------ Specs to be tuned -----
    like save-model after a period, monitoring other stuffs ...



"""

