# -*- coding: utf-8 -*-
import gzip
from ..utils.data_utils import get_file
from ..utils.data_utils import get_file_from_dir
from six.moves import cPickle
import sys

# Read the original mnist data file.
import struct
from numpy import append, array, int8, uint8, zeros, arange
from array import array as pyarray


def load_data(path="mnist.pkl.gz"):
    tpath = get_file(path, origin="https://s3.amazonaws.com/img-datasets/mnist.pkl.gz")
    # loadcompleteimages from locsal file.

    # tpath = get_file_from_dir(path, directory='/Users/kyu/BaiduYun/Dropbox/git/keras/data')
    if tpath is None:
        path = get_file_from_dir(path, directory='/home/kyu/Dropbox/git/keras/data')
    else:
        path = tpath
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    if sys.version_info < (3,):
        data = cPickle.loadcompleteimages(f)
    else:
        data = cPickle.loadcompleteimages(f, encoding="bytes")

    f.close()
    # return data  # (X_train, y_train), (X_test, y_test)
    return data # (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def load_infimnist(name='mnist60k', pattern_name='', label_name='', digits=arange(10)):
    """
    Load infimnist result
    Reference: Gustav, http://g.sweyla.com/blog/2012/mnist-numpy/
    :param name:
    :param pattern_name:
    :param label_name:
    :param digits:
    :return:
    """
    if pattern_name is '':
        pattern_name = '-patterns-idx3-ubyte'
    if label_name is '':
        label_name = '-labels-idx1-ubyte'
    data_dir = 'infimnist/data/'
    pattern = get_file(data_dir + name + pattern_name, origin=None)
    label = get_file(data_dir + name + label_name, origin=None)

    # Read Train image
    pat_file = open(pattern, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", pat_file.read(16))
    pat = pyarray("B", pat_file.read())
    pat_file.close()

    # Read Train label
    lab_file = open(label, 'rb')
    magic_nr, size = struct.unpack(">II", lab_file.read(8))
    lab = pyarray("B", lab_file.read())
    lab_file.close()

    # Transform into numpy array
    # Record the index
    index = [k for k in range(size) if lab[k] in digits]
    N = len(index)
    np_pat = zeros((N, rows, cols), dtype=uint8)
    np_lab = zeros((N,), dtype=int8)
    for i in range(N):
        np_pat[i] = array(pat[index[i]*rows*cols: (index[i]+1)*rows*cols]).reshape((rows, cols))
        np_lab[i] = lab[index[i]]
    # Return readed result.
    return np_pat, np_lab