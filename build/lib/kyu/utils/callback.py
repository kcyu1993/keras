import warnings
import keras.backend as K
from keras.callbacks import Callback
import numpy as np


class ReduceLROnDemand(Callback):
    '''Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    # Example
        ```python
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=5, min_lr=0.001)
            model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```

    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    '''

    def __init__(self, factor=0.1, sequence=10,
                 verbose=1, epsilon=1e-4, min_lr=0):
        super(Callback, self).__init__()
        if factor >= 1.0:
            raise ValueError('ReduceOnDemand does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        if isinstance(sequence, int):
            # assume the number of epoch will not exceed 10000
            self.sequence = range(int(sequence), int(1000), int(sequence))
        elif isinstance(sequence, (list,tuple)):
            self.sequence = sequence
        else:
            raise RuntimeError("Sequence must be either a int or list/tuple")
        print("Reduced on demand :" + str(self.sequence))
        self.verbose = verbose

        self.monitor_op = None
        self.reset()

    def reset(self):
        self.lr_epsilon = self.min_lr * 1e-4

    def on_train_begin(self, logs={}):
        self.reset()

    def on_epoch_end(self, epoch, logs={}):
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        if epoch in self.sequence:
            old_lr = float(K.get_value(self.model.optimizer.lr))
            if old_lr > self.min_lr + self.lr_epsilon:
                new_lr = old_lr * self.factor
                new_lr = max(new_lr, self.min_lr)
                K.set_value(self.model.optimizer.lr, new_lr)
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
