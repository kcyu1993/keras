from keras import backend as K
from keras.optimizers import Optimizer


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