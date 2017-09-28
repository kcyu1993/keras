from keras.regularizers import Regularizer
import keras.backend as K

def robust_estimate_eigenvalues(s, alpha):
    """
    Robust estimatation in RAID-G paper


    Reference
    ----------
        Wang, Q., Li, P., Zuo, W., & Zhang, L. (2016).
        RAID-G - Robust Estimation of Approximate Infinite Dimensional
            Gaussian with Application to Material Recognition.

    Parameters
    ----------
    s : tf.tensor   Tensorflow input

    Returns
    -------

    """
    return K.sqrt(K.pow((1 - alpha) / 2 / alpha, 2) + s / alpha) - (1-alpha) / (2*alpha)


class FrobNormRegularizer(Regularizer):
    def __init__(self, dim, alpha=0.01):
        self.alpha = alpha
        self.dim = dim

    def __call__(self, x):
        regularization = self.alpha * K.sqrt(K.sum((x - K.eye(self.dim))**2)) / self.dim
        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'dim': float(self.dim),
                'alpha': float(self.alpha)}


class VonNeumannDistanceRegularizer(Regularizer):
    def __init__(self, dim, alpha=0.01, epsilon=1e-5):
        self.alpha = alpha
        self.dim = dim
        self.eps = epsilon
        if K.backend() == 'theano':
            raise RuntimeError("v-N divergence not support theano now")

    def __call__(self, x):
        """
        Define the regularization of von Neumann matrix divergence

        Parameters
        ----------
        x

        Returns
        -------

        """
        import tensorflow as tf
        s, u = tf.self_adjoint_eig(x)
        ## TO ensure the stability
        comp = tf.zeros_like(s) + self.eps
        # comp = tf.Print(comp, [comp], message='comp')
        inner = tf.where(tf.less(s, comp), comp, s)
        # inner = tf.Print(inner, [inner], message='inner', summarize=self.dim)
        inner = tf.log(inner)
        von = tf.matmul(tf.matrix_diag(s), tf.matrix_diag(inner)) - tf.matrix_diag(s - 1)
        von = tf.matmul(u, tf.matmul(von, tf.transpose(u, [0,2,1])))
        # von = tf.Print(von, [von], message='von')
        reg = tf.reduce_sum(self.alpha * tf.trace(von, 'vN_reg')) / self.dim
        # reg = tf.Print(reg, [reg], message='vN_reg')
        return reg

    def get_config(self):
        return {'name': self.__class__.__name__,
                'dim': float(self.dim),
                'alpha': float(self.alpha),
                }


def fob(dim, alpha=0.01):
    return FrobNormRegularizer(dim, alpha)


def vN(dim, alpha=0.01, epsilon=1e-5):
    return VonNeumannDistanceRegularizer(dim, alpha, epsilon)