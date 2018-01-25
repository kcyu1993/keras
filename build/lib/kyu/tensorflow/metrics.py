"""
Add custom metric

"""

import keras.backend as K


def index_binary_accuracy(y_true, y_pred, index):
    """
    Give the index, compute the binary accuracy for exactly one category

    Parameters
    ----------
    y_true : shape (Batches, dim)
    y_pred : shape (Batches, dim)
    index : 0 < ind < dim

    Returns : shape(Batches, ) as accuracy
    -------

    """
    pred_shape = K.int_shape(y_pred)
    # assert pred_shape == K.int_shape(y_true)
    if index >= pred_shape[-1]:
        raise ValueError("Index must be smaller than prediction shape dimension")

    # cast with index and return the binary accuracy, on the cut of 0.5.
    return K.cast(K.equal(y_true[:, index],
                          K.round(y_pred[:, index])),
                  K.floatx())


class IndexBinaryAccuracy(object):
    def __init__(self, index=0, label=None):
        self.index = index
        self.__name__ = '{}_acc'.format(index if label is None else label)

    def __call__(self, y_true, y_pred):
        index = self.index
        shape = K.int_shape(y_pred)
        # assert pred_shape == K.int_shape(y_true)
        if index >= shape[-1]:
            raise ValueError("Index must be smaller than prediction shape dimension")

        # cast with index and return the binary accuracy, on the cut of 0.5.
        return K.cast(K.equal(y_true[:, index],
                              K.round(y_pred[:, index])),
                      K.floatx())


class ConfusionMatrix(object):
    def __init__(self):
        self.__name__ = "confusion_mat"

    def __call__(self, y_true, y_pred):
        from tensorflow.contrib.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)
