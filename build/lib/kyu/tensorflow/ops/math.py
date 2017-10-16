"""
Implement math operations

"""
import tensorflow as tf
from tensorflow.python.framework.function import Defun

import keras.backend as K
from keras.optimizers import SGD, Optimizer
from kyu.tensorflow.ops.svd_gradients import get_eigen_K

_EPSILON = 1e-4


def gradient_matrix_eig_safe(op, grad_s, grad_u):
    """
    Implement the safe gradients for EIG.
        Double precision operations for EIG safe computation.

    Parameters
    ----------
    op
    grad_s
    grad_u

    Returns
    -------

    """
    with tf.device("/cpu:0"):
        dtype = tf.float64
        s, u = op.outputs
        s = tf.identity(s, name='s')
        u = tf.identity(u, name='u')

        s = tf.cast(s, dtype, name='s_float64')
        u = tf.cast(u, dtype, name='u_float64')
        grad_s = tf.cast(grad_s, dtype, name='grad_s_float64')
        grad_u = tf.cast(grad_u, dtype, name='grad_u_float64')

        # Compute the s1, u1 which has non-zero values.

        u_t = tf.transpose(u, [0,2,1])
        with tf.name_scope('K'):
            K = get_eigen_K(s, square=False, dtype=dtype)

        with tf.name_scope('mask_K'):
            """ Remove the o """
            # mask_K = tf.where(tf.greater(K, tf.zeros_like(K) + _EPSILON),
            #                   tf.ones_like(K), tf.zeros_like(K))
            mask_s = tf.where(tf.greater(s, tf.zeros_like(s, dtype=dtype) + _EPSILON),
                              tf.ones_like(s, dtype=dtype), tf.zeros_like(s, dtype=dtype))
            mask_s = tf.expand_dims(mask_s, -1)
            mask_K = tf.matmul(mask_s, tf.transpose(mask_s, [0, 2, 1]))

        with tf.name_scope('inner'):
            inner = K * tf.matmul(u_t, grad_u) + tf.matrix_diag(grad_s)

        with tf.name_scope('inner_masked'):
            inner *= mask_K

        with tf.name_scope('dzdx_final'):
            dzdx_final = tf.matmul(inner, u_t)
            dzdx_final = tf.identity(tf.matmul(u, dzdx_final), name='identity')

        with tf.name_scope('dzdx_wrapper'):
            dzdx_final = tf.where(tf.is_nan(dzdx_final), tf.zeros_like(dzdx_final, dtype=dtype), dzdx_final)
        dzdx_final = tf.cast(dzdx_final, tf.float32, name='dzdx_final_float32')
    return dzdx_final


def safe_matrix_eig(x):
    """
    Lets return the U1 and S1 accordingly, rather than the complete matrix.

    Parameters
    ----------
    x

    Returns
    -------

    """
    return tf.self_adjoint_eig(x)


@Defun(tf.float32, python_grad_func=gradient_matrix_eig_safe)
def safe_matrix_eig_op(x):
    """ TensorFlow Safe EIG op definition """
    dtype = tf.float32
    x = tf.cast(x, dtype=dtype)
    # with tf.device('/cpu:0'):
    s, u = safe_matrix_eig(x)
    s, u = tf.cast(s, tf.float32, name='s_float32'), tf.cast(u, tf.float32, name='u_float32')
    return s, u


def matrix_exp(x):
    s, u = tf.self_adjoint_eig(x)
    # s = tf.abs(s)
    inner = s
    inner = tf.exp(inner)
    inner = tf.where(tf.is_nan(inner), tf.zeros_like(inner), inner)
    inner = tf.matrix_diag(inner)
    tf_exp = tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0, 2, 1])))
    return tf_exp


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
        p_shape = new_p.get_shape()
        if p_shape[0]._value > p_shape[1]._value:
            new_p, _ = tf.qr(new_p, full_matrices=False)
        else:
            new_p, _ = tf.qr(tf.transpose(new_p), full_matrices=False)
            new_p = tf.transpose(new_p)
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

class TFOptimizer_v2(Optimizer):
    """
    Wrapper class for native TensorFlow optimizers.
        Support the Horovod distributed optimizer

    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        # if isinstance(optimizer, tf.train.Optimizer):
        #     self.lr = optimizer._learning_rate
        # else:
        # Assume its the horovod opt
        opt = optimizer._optimizer
        if hasattr(opt, '_lr'):
            lr = getattr(opt, '_lr')
        elif hasattr(opt, '_learning_rate'):
            lr = getattr(opt, '_learning_rate')
        else:
            raise ValueError
        self.lr = K.variable(lr, name='lr')

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    def get_updates(self, loss, params):
        grads = self.optimizer.compute_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        opt_update = self.optimizer.apply_gradients(
            grads, global_step=self.iterations)
        self.updates.append(opt_update)
        return self.updates

    @property
    def weights(self):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    def from_config(self, config):
        raise NotImplementedError
