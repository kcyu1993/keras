import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test, keras_test
from keras import backend as K
from keras.layers import SecondaryStatistic, WeightedProbability, O2Transform



@keras_test
def test_secondstat():
    nb_samples = 2
    nb_steps = 8
    cols = 10
    rows = 12
    nb_filter = 3
    input_shape = (nb_samples, nb_filter, cols, rows)
    layer_test(SecondaryStatistic,
               kwargs={
                   'output_dim':None,
                   'parametrized':False,
                   'init':'glorot_uniform',
                   'activation':'linear',
               },
               input_shape=input_shape)


@keras_test
def test_WeightProbability():
    raise NotImplementedError


@keras_test
def test_O2Transform():
    raise NotImplementedError

if __name__ == '__main__':
    # TODO Finish the testing cases for self defined layers
    pytest.main([__file__])