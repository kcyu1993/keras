"""
Define image loading related operations

"""

from PIL import Image
import os
import numpy as np

import keras.backend as K
from keras.preprocessing.image import Iterator, load_img, img_to_array, array_to_img, ImageDataGenerator


class ImageLoader(object):

    def __init__(self, dirpath):
        """Create General ImageLoader related operations """
        self.r_dirpath = dirpath

        self.categories = None
        self.label = []
        self.image = []

        self.image_dir = None
        self.label_dir = None


    def _load(self, fpath):
        """
                Load single image from given path
                :param fpath: full path.
                :return:      class label, image 3D array
                """
        dir, filename = os.path.split(fpath)
        index, _ = self.decode(filename)
        fpath = self.getimagepath(fpath)
        img = ImageLoader.load_image(fpath)
        return index, img

    @staticmethod
    def load_image(fpath):
        """
        Load the jpg image as np.ndarray(3, row, col)
        Following the theano image ordering.

        :param fpath:
        :return:
        """
        img = Image.open(fpath)
        img.load()
        data = np.asarray(img, dtype="byte")
        if len(data.shape) == 3:
            data = data.transpose([2, 0, 1])
        elif len(data.shape) == 2:
            data = np.ones((3, data.shape[0], data.shape[1])) * data
        return data

    """ Abstract methods """
    def getimagepath(self, fpath):
        pass

    def decode(self, filename):
        pass


