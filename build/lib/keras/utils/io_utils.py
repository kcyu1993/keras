from __future__ import absolute_import
from __future__ import print_function

import gzip
import os
import numpy as np
import sys
from collections import defaultdict
from six.moves import cPickle

class HDF5Matrix():
    '''Representation of HDF5 dataset which can be used instead of a
    Numpy array.

    # Example

    ```python
        X_data = HDF5Matrix('input/file.hdf5', 'data')
        model.predict(X_data)
    ```

    Providing start and end allows use of a slice of the dataset.

    Optionally, a normalizer function (or lambda) can be given. This will
    be called on every slice of data retrieved.

    # Arguments
        datapath: string, path to a HDF5 file
        dataset: string, name of the HDF5 dataset in the file specified
            in datapath
        start: int, start of desired slice of the specified dataset
        end: int, end of desired slice of the specified dataset
        normalizer: function to be called on data when retrieved

    '''
    refs = defaultdict(int)

    def __init__(self, datapath, dataset, start=0, end=None, normalizer=None):
        import h5py

        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath)
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.data = f[dataset]
        self.start = start
        if end is None:
            self.end = self.data.shape[0]
        else:
            self.end = end
        self.normalizer = normalizer

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.stop + self.start <= self.end:
                idx = slice(key.start+self.start, key.stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, int):
            if key + self.start < self.end:
                idx = key + self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
            else:
                raise IndexError
        if self.normalizer is not None:
            return self.normalizer(self.data[idx])
        else:
            return self.data[idx]

    @property
    def shape(self):
        return (self.end - self.start,) + self.data.shape[1:]


def save_array(array, name):
    import tables
    f = tables.open_file(name, 'w')
    atom = tables.Atom.from_dtype(array.dtype)
    ds = f.createCArray(f.root, 'data', atom, array.shape)
    ds[:] = array
    f.close()


def load_array(name):
    import tables
    f = tables.open_file(name)
    array = f.root.data
    a = np.empty(shape=array.shape, dtype=array.dtype)
    a[:] = array[:]
    f.close()
    return a


def ask_to_proceed_with_overwrite(filepath):
    get_input = input
    if sys.version_info[:2] <= (2, 7):
        get_input = raw_input
    overwrite = get_input('[WARNING] %s already exists - overwrite? '
                          '[y/n]' % (filepath))
    while overwrite not in ['y', 'n']:
        overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
    if overwrite == 'n':
        return False
    print('[TIP] Next time specify overwrite=True!')
    return True


def cpickle_load(filename):
    """
    Support loading from files directly.
    Proved to be same speed, but large overhead.
    Switch to load generator.
    :param filename:
    :return:
    """
    if os.path.exists(filename):
        if filename.endswith(".gz"):
            f = gzip.open(filename, 'rb')
        else:
            f = open(filename, 'rb')
        # f = open(path, 'rb')
        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding="bytes")
        f.close()
        return data  # (data
    else:
        print("File not found under location {}".format(filename))
        return None


def cpickle_save(data, output_file, ftype='gz'):
    """
    Save the self.label, self.image before normalization, to Pickle file
    To the specific output directory.
    :param output_file:
    :param ftype
    :return:
    """
    if ftype is 'gz' or ftype is 'gzip':
        print("compress with gz")
        f = gzip.open((output_file + "." + ftype), 'wb')
    elif ftype is '':
        f = open((output_file + "." + ftype), 'wb')
    else:
        raise ValueError("Only support type as gz or '' ")
    cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()