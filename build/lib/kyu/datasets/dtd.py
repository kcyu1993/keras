"""
Described Texture Dataset


"""
from __future__ import absolute_import
from __future__ import print_function

from kyu.engine.utils.data_utils import ClassificationImageData
from kyu.utils.dict_utils import create_dict_by_given_kwargs

from kyu.utils.image import ClassificationIterator

from scipy.io import loadmat
from kyu.utils.io_utils import get_dataset_dir
import numpy as np
from os import path as path

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

    y_train -= 1
    y_valid -= 1
    y_test -= 1
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


class DTD(ClassificationImageData):

    def __init__(self, root_folder, use_validation=False, image_dir=IMAGE_PATH, name='DTD', meta_folder='imdb'):
        self.meta_file = path.join(root_folder, meta_folder, 'imdb.mat')
        self.mat = self.load_mat()
        super(DTD, self).__init__(root_folder=root_folder, image_dir=image_dir, name=name, meta_folder=meta_folder,
                                  use_validation=use_validation)
        # Construct the image label list
        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[244, 296], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=256, random_crop=False, horizontal_flip=True)

    def load_mat(self):
        meta_path = self.meta_file
        mat = loadmat(meta_path)
        return mat

    def decode(self, path):
        """ decode the image path info to string """
        raise NotImplementedError

    def build_image_label_lists(self):
        images = self.mat['images']

        # images_set = images['set']
        # images_name = images['name'][0, 0]

        # Translate image names into image locations
        # images_loc = [unicode_nparray_to_str(images_name[0,i]) for i in range(0,images_name.shape[1])]

        train_name = images['name'][0, 0][np.where(images['set'][0, 0] == 1)]
        val_name = images['name'][0, 0][np.where(images['set'][0, 0] == 2)]
        if not self.use_validation:
            train_name = np.append(train_name, val_name)
        test_name = images['name'][0, 0][np.where(images['set'][0, 0] == 3)]

        y_train = images['class'][0, 0][np.where(images['set'][0, 0] == 1)]
        y_valid = images['class'][0, 0][np.where(images['set'][0, 0] == 2)]
        if not self.use_validation:
            y_train = np.append(y_train, y_valid)
        y_test = images['class'][0, 0][np.where(images['set'][0, 0] == 3)]

        y_train -= 1
        y_valid -= 1
        y_test -= 1

        self._set_train(
            [path.join(self.image_folder, file_name[0]) for file_name in train_name],
            y_train,
        )

        if self.use_validation:
            self._set_valid(
                [path.join(self.image_folder, file_name[0]) for file_name in val_name],
                y_valid,
            )

        self._set_test(
            [path.join(self.image_folder, file_name[0]) for file_name in test_name],
            y_test,
        )

    def _build_category_dict(self):
        """ Build the category dictionary by meta-file """
        meta = self.mat['meta']
        classes = meta['classes'][0][0][0]
        # To python str for unified usage
        classes_names = [str(classes[i][0]) for i in range(len(classes))]
        category_dict = dict(zip(classes_names, range(len(classes_names))))
        self.nb_class = len(classes_names)
        return category_dict
