"""
Define image loading related operations

"""

from PIL import Image
import os
import numpy as np

import keras.backend as K
from keras.preprocessing.image import Iterator, load_img, img_to_array, array_to_img, ImageDataGenerator


def get_project_dir():
    """
    Get the current project directory
    :return:
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
    # current_dir = os.path.dirname(current_dir)
    # current_dir = os.path.dirname(current_dir)
    return current_dir


def get_dataset_dir():
    """
    Get current dataset directory
    :return:
    """
    # current_dir = get_project_dir()
    # return os.path.join(current_dir, 'data')
    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    return os.path.join(datadir_base, 'datasets')

def get_absolute_dir_project(filepath):
    """
    Get the absolute dir path under project folder.
    :param dir_name: given dir name
    :return: the absolute path.
    """
    path = os.path.join(get_project_dir(), filepath)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return path


def get_weight_path(filename, dir='project'):
    if dir is 'project':
        path = get_absolute_dir_project('model_saved')
        return os.path.join(path, filename)
    elif dir is 'dataset':
        dir_base = os.path.expanduser(os.path.join('~', '.keras'))
        dir = os.path.join(dir_base, 'models')
        if not os.path.exists(dir):
            os.mkdir(dir)
        return os.path.join(dir, filename)


def get_plot_path(filename, dir='project'):
    if dir is 'project':
        path = get_absolute_dir_project('model_saved/plots')
        if not os.path.exists(path):
            os.mkdir(path)
    elif dir is 'dataset':
        dir_base = os.path.expanduser(os.path.join('~', '.keras'))
        path = os.path.join(dir_base, 'plots')
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        raise ValueError("Only support project and dataset as dir input")
    if filename == '' or filename is None:
        return path
    return os.path.join(path, filename)


def get_plot_path_with_subdir(filename, subdir, dir='project'):
    """
    Allow user to get plot path with subdir

    Parameters
    ----------
    filename
    subdir : str    support
        'summary' for summary graph, 'run' for run-time graph, 'models' for model graph
    dir : str       support 'project' under project plot path, 'dataset' under ~/.keras/plots

    Returns
    -------
    path : str      absolute path of the given filename, or directory if filename is None or ''
    """
    path = get_plot_path('', dir=dir)
    path = os.path.join(path, subdir)
    if not os.path.exists(path):
        os.mkdir(path)
    if filename == '' or filename is None:
        return path
    return os.path.join(path, filename)


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


