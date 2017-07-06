# import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.layers import Dense, Activation, Input, Reshape
from keras.utils.test_utils import layer_test, keras_test
from kyu.tensorflow.ops import normalization
from keras.models import Sequential, Model
from keras import backend as K

input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones((10))
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


def get_covariance_matrices(batch_size, nb_channel, dim, rank):
    input_shape = (batch_size, nb_channel, dim, rank)
    data = np.random.randn(*input_shape).astype(K.floatx())
    mean = np.mean(data, axis=3, keepdims=True)
    data_norm = data - mean
    cov = np.zeros((batch_size, nb_channel, dim, dim))
    for i in range(batch_size):
        for j in range(nb_channel):
            cov[i,j,:,:] = np.cov(data_norm[i,j,:,:], rowvar=True)
            # print(cov[i,j,:,:].shape)
    cov = np.transpose(cov, [0,2,3,1])
    if nb_channel == 1:
        cov = np.squeeze(cov)
    return cov


@keras_test
def basic_batchnorm_test():
    from keras import regularizers
    layer_test(normalization.SecondOrderBatchNormalization,
               kwargs={'mode': 1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2))
    layer_test(normalization.SecondOrderBatchNormalization,
               kwargs={'mode': 0},
               input_shape=(3, 4, 2))


@keras_test
def test_secondbatchnorm_mode_0_or_2():
    for mode in [0, 2]:
        model = Sequential()
        norm_m0 = normalization.SecondOrderBatchNormalization(so_mode=0, mode=mode, input_shape=(10,), momentum=0.8)
        # norm_m0 = normalization.BatchNormalization(mode=mode, input_shape=(10,), momentum=0.8)
        model.add(norm_m0)
        model.compile(loss='mse', optimizer='sgd')

        # centered on 5.0, variance 10.0
        X = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
        model.fit(X, X, nb_epoch=4, verbose=0)
        out = model.predict(X)
        # print(" {} {} ".format(K.eval(norm_m0.beta) , K.eval(norm_m0.gamma)) )
        out -= K.eval(norm_m0.beta)
        out /= K.eval(norm_m0.gamma)

        assert_allclose(out.mean(), 0.0, atol=1e-1)
        assert_allclose(out.std(), 1.0, atol=1e-1)


@keras_test
def test_batchnorm_mode_0_or_2_twice():
    # This is a regression test for issue #4881 with the old
    # batch normalization functions in the Theano backend.
    model = Sequential()
    model.add(normalization.SecondOrderBatchNormalization(so_mode=0, mode=0, input_shape=(10, 5, 5), axis=1))
    model.add(normalization.SecondOrderBatchNormalization(so_mode=0, mode=0, input_shape=(10, 5, 5), axis=1))
    model.compile(loss='mse', optimizer='sgd')

    X = np.random.normal(loc=5.0, scale=10.0, size=(20, 10, 5, 5))
    model.fit(X, X, nb_epoch=1, verbose=0)
    model.predict(X)


@keras_test
def test_sobatchnorm_so_mode_1():
    model = Sequential()
    norm_m0 = normalization.SecondOrderBatchNormalization(so_mode=1, mode=0, input_shape=(5, 5), momentum=0.8, axis=-1)
    model.add(norm_m0)
    # model.add(normalization.SecondOrderBatchNormalization(so_mode=1, mode=0, input_shape=(5, 5), axis=0))
    model.compile(loss='mse', optimizer='sgd')
    # X = np.random.normal(loc=5, scale=10, size=(10, 5, 5))
    # X = np.matmul
    X = get_covariance_matrices(10, 1, 5, 8)
    # X = X.reshape(10, -1)
    model.fit(X, X, nb_epoch=12, verbose=0)
    out = model.predict(X)
    out -= K.eval(norm_m0.beta)
    out /= K.eval(norm_m0.gamma)

    assert_allclose(out.mean(), 0.0, atol=1e-1)
    assert_allclose(out.std(), 1.0, atol=1e-1)

@keras_test
def test_secondorderbatchnorm_so_mode_1vs0(nb_epoch=10):
    model1 = Sequential()

    norm_m0 = normalization.SecondOrderBatchNormalization(
        so_mode=0, mode=0, momentum=0.8, axis=-1, input_shape=(5,5)
    )
    # model1.add(Reshape([25,], input_shape=(5,5)))
    model1.add(norm_m0)
    # model1.add(Reshape([5, 5]))
    model1.compile(loss='mse', optimizer='sgd')

    model2 = Sequential()
    norm_m1 = normalization.SecondOrderBatchNormalization(
        so_mode=1, mode=0, input_shape=(5, 5), momentum=0.8, axis=-1
    )
    model2.add(norm_m1)
    model2.compile(loss='mse', optimizer='sgd')

    X = get_covariance_matrices(100, 1, 5, 8)
    model1.fit(X, X, nb_epoch=nb_epoch, verbose=0)
    model2.fit(X, X, nb_epoch=nb_epoch, verbose=0)

    out1 = model1.predict(X)
    out2 = model2.predict(X)

    assert_allclose(out1, out2, atol=1e-1)


@keras_test
def test_secondorderbatchnorm_so_mode_2(nb_epoch=10):
    model = Sequential()
    norm_m0 = normalization.SecondOrderBatchNormalization(
        so_mode=2, mode=0, momentum=0.8, axis=-1, input_shape=(5,5)
    )
    model.add(norm_m0)
    model.compile(loss='mse', optimizer='sgd')
    X = get_covariance_matrices(100, 1, 5, 8)
    model.fit(X, X, nb_epoch=nb_epoch, verbose=0)
    out = model.predict(X)

    assert_allclose(out.mean(), 0.0)
    assert_allclose(out.std(), 1.0)


if __name__ == '__main__':
    # pytest.main([__file__])
    # basic_batchnorm_test()
    # test_batchnorm_mode_0_or_2_twice()
    # test_secondorderbatchnorm_so_mode_1vs0(200)
    # test_sobatchnorm_so_mode_1()
    test_secondorderbatchnorm_so_mode_2(100)