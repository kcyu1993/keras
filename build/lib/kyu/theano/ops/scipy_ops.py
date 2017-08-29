from __future__ import absolute_import, print_function
import logging
import warnings
from six.moves import xrange

# Theano related import
import numpy
from theano import tensor
import theano.tensor
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply

# Try to import scipy
try:
    import scipy.linalg
    imported_scipy = True

except ImportError:
    imported_scipy = False


logger = logging.getLogger(__name__)

class Logm(Op):
    """
    Compute the matrix logarithm of a square array, and extented to 4-D array

    """

    __props__ = ()

    def make_node(self, A):
        assert imported_scipy, (
            "Scipy not avaliable, Scipy neede for the Logm op")
        A = as_tensor_variable(A)
        assert A.ndim == 2
        logm = theano.tensor.matrix(dtype=A.dtype)
        return Apply(self, [A,], [logm,])

    def perform(self, node, inputs, outputs):
        (A,) =inputs
        (logm,) = outputs
        logm[0] = scipy.linalg.logm(A)

    def grad(self, inputs, outputs):
        (A,) = inputs
        (g_out,) = outputs
        return [LogmGrad()(A, g_out)]

    def infer_shape(self, node, shapes):
        return [shapes[0]]


class LogmGrad(Op):
    """
    Gradient of matrix logarithm of a square array.

    """

    __props__ = ()

    def make_node(self, A, gw):
        assert imported_scipy, (
            "Scipy not avaliable, Scipy neede for the Logm op")
        A = as_tensor_variable(A)
        assert A.ndim == 2
        out = theano.tensor.matrix(dtype=A.dtype)
        return Apply(self, [A, gw], [out, ])

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        # Kalbfleisch and Lawless, J. Am. Stat. Assoc. 80 (1985) Equation 3.4
        # Kind of... You need to do some algebra from there to arrive at
        # this expression.
        (A, gA) = inputs
        (out,) = outputs
        w, V = scipy.linalg.eig(A, right=True)
        U = scipy.linalg.inv(V).T

        exp_w = numpy.exp(w)
        X = numpy.subtract.outer(exp_w, exp_w) / numpy.subtract.outer(w, w)
        numpy.fill_diagonal(X, exp_w)
        Y = U.dot(V.T.dot(gA).dot(U) * X).dot(V.T)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", numpy.ComplexWarning)
            out[0] = Y.astype(A.dtype)
