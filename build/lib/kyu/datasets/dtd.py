"""
Described Texture Dataset


"""
from __future__ import absolute_import
from __future__ import print_function

from keras.preprocessing.image import ClassificationIterator

from scipy.io import loadmat
from keras.utils.data_utils import get_dataset_dir
import numpy as np

ROOT_PATH = get_dataset_dir() + '/dtd'
IMAGE_PATH = ROOT_PATH + '/images'


def load_mat():
    matpath = ROOT_PATH + '/imdb/imdb.mat'
    mat = loadmat(matpath)
    return mat


def unicode_nparray_to_str(uni_np):
    try:
        return uni_np.astype(str)[0]
    except Exception:
        return None


def decode_mat():
    mat = load_mat()

    meta = mat['meta']
    classes = meta['classes'][0][0]
    classes = classes.tolist()[0]   # Transform to list

    images = mat['images']


    images_set = images['set']
    images_name = images['name'][0,0]

    # Translate image names into image locations
    # images_loc = [unicode_nparray_to_str(images_name[0,i]) for i in range(0,images_name.shape[1])]

    train_name = images['name'][0,0][np.where(images['set'][0,0] == 1)]
    val_name = images['name'][0,0][np.where(images['set'][0,0] == 2)]
    test_name = images['name'][0,0][np.where(images['set'][0,0] == 3)]

    y_train = images['class'][0,0][np.where(images['set'][0,0] == 1)]
    y_valid = images['class'][0,0][np.where(images['set'][0,0] == 2)]
    y_test = images['class'][0,0][np.where(images['set'][0,0] == 3)]

    y_train = y_train - 1
    y_valid= y_valid - 1
    y_test = y_test - 1
    return train_name, val_name, test_name, y_train, y_valid, y_test


def load_dtd(no_valid=True, image_gen=None, batch_size=32):
    """
    Load dtd as ClassificationIterator



    Returns
    -------
    train, valid, test as ClassificationIterator
    """
    train_name, val_name, test_name, y_train, y_valid, y_test = decode_mat()
    train_loc = [unicode_nparray_to_str(train_name[i]) for i in range(0, train_name.shape[0])]
    valid_loc = [unicode_nparray_to_str(val_name[i]) for i in range(0, val_name.shape[0])]
    test_loc = [unicode_nparray_to_str(test_name[i]) for i in range(0, test_name.shape[0])]

    if no_valid:
        train_loc += valid_loc
        y_train = np.concatenate((y_train, y_valid), axis=0)
        train = ClassificationIterator(train_loc, y_train, image_gen, load_path_prefix=IMAGE_PATH, batch_size=batch_size)
        valid = None

    else:
        train = ClassificationIterator(train_loc, y_train, image_gen, load_path_prefix=IMAGE_PATH, batch_size=batch_size)
        valid = ClassificationIterator(valid_loc, y_valid, image_gen, load_path_prefix=IMAGE_PATH, batch_size=batch_size)

    test = ClassificationIterator(test_loc, y_test, image_gen, load_path_prefix=IMAGE_PATH, batch_size=batch_size)

    return train, valid, test
