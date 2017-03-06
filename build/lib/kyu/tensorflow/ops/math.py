"""
Implement math operations

"""
import tensorflow as tf
import keras.backend as K
from keras.optimizers import SGD


def matrix_sym_op(x):
    """
    Defined for Sym(A) = (A + A')/2

    Parameters
    ----------
    x

    Returns
    -------

    """
    return (x + tf.transpose(x))/2


def get_matrix_norm(x, mode='Frob', keep_dim=False):
    """
    Tensorflow Frob norm

    Parameters
    ----------
    x : Tensor with shape [..., M, M]
    mode : string Frob at the time being.

    Returns
    -------
    tf.scalar
    """

    if mode == 'Frob':
        if keep_dim:
            return tf.sqrt(tf.trace(tf.matmul(x, tf.transpose(x, [0,2,1]))))
        else:
            return tf.reduce_sum(tf.pow(x, tf.ones_like(x) * 2))
    else:
        raise RuntimeError("Not supported norm: " + mode)


class StiefelSGD(SGD):

    def __init__(self, lr=0.01, momentum=0.9, decay=1e-6,
                 nesterov=False,
                 observed_names=None,
                 **kwargs):
        """
        Stiefle-SGD
        would support specific layer to do Stiefel-updates.

        Parameters
        ----------
        lr
        momentum
        decay
        nesterov
        kwargs
        """
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)
        self.inital_decay = decay

        # For configuration purpose.
        self.init_lr = lr
        self.init_momentum = momentum
        # Class name to the observed layer.
        self.observed_names = observed_names

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        for p, g, m in zip(params, grads, moments):

            if StiefelSGD.find_obversved_name(p.name, self.observed_names):
                self.stiefel_update(p, constraints, g, m, lr)
                continue

            # For some group, you do normal
            v = self.momentum * m - lr * g  # velocity
            # v.name = p.name + '_v'
            # m.name = p.name + '_m'
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def stiefel_update(self, param, constraints, grad, moment, lr):
        """
        Override the Stiefel updates accordingly.

        Parameters
        ----------
        param
        constraint
        grad

        Returns
        -------
        corresponding Stiefel updates.
        """
        p = param
        m = moment
        g = grad

        new_g = g - K.dot(p, matrix_sym_op(K.dot(K.transpose(p), g)))

        v = self.momentum * m - lr * new_g  # velocity
        # v.name = p.name + '_v'
        # m.name = p.name + '_m'
        self.updates.append(K.update(m, v))


        # if self.nesterov:
        #     new_p = p + self.momentum * v - lr * new_g
        # else:
        new_p = p + v
        new_p, _ = tf.qr(new_p, full_matrices=False)
        # apply constraints
        if p in constraints:
            c = constraints[p]
            new_p = c(new_p)

        self.updates.append(K.update(p, new_p))


    @staticmethod
    def find_obversved_name(name, keywords):
        """
        Find if the name contains the observed names

        Parameters
        ----------
        name
        keywords

        Returns
        -------
        Boolean
        """
        for key in keywords:
            if key in name:
                return True
        return False