from keras.regularizers import Regularizer
import keras.backend as K


class FobNormRegularizer(Regularizer):
    def __init__(self, dim, alpha=0.01):
        self.alpha = alpha
        self.dim = dim

    def __call__(self, x):
        regularization = self.alpha * K.sum((x - K.eye(self.dim))**2)
        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'dim': float(self.dim),
                'alpha': float(self.alpha)}


def fob(dim, alpha=0.01):
    return FobNormRegularizer(dim, alpha)
