import pytest
import os
import numpy as np

from keras.utils.test_utils import get_test_data
from keras.utils import visualize_util

epoch = 100
tr = (10 + np.random.rand(epoch), np.random.rand(epoch))
te = (10 + np.random.rand(epoch), np.random.rand(epoch))


def test_plot_train_test():

    path = visualize_util.plot_train_test(tr[0], tr[1], range(epoch),
                                   xlabel='epoch', ylabel='loss',
                                   filename='test.png', plot_type=0)
    print(path)
    assert os.path.exists(path)
    # os.remove(path)

def test_multiple_model():
    tr = (np.random.rand(epoch),np.random.rand(epoch),np.random.rand(epoch))
    te = (np.random.rand(epoch),np.random.rand(epoch),np.random.rand(epoch))
    modelnames=('model1', 'model2', 'model3')
    path = visualize_util.plot_multiple_train_test(tr, te, modelnames, range(epoch),
                                                   xlabel='epoch', ylabel='acc',
                                                   filename='test3.png')
    print(path)
    assert os.path.exists(path)
    # os.remove(path)

def test_plot_multi_train_test():

    path = visualize_util.plot_loss_acc(
        tr[0], te[0], tr[1], te[1], range(epoch),
        xlabel='epoch', ylabel=['loss','acc'],
        linestyle=['-', '-.'],
        filename='test2.png')
    print(path)
    assert os.path.exists(path)
    # os.remove(path)



if __name__ == '__main__':
    pytest.main([__file__])
