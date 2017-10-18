"""
Nested the ImageIterator and the data meta data together.

"""
from abc import abstractmethod

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from kyu.utils.image import ImageIterator, ImageDataGeneratorAdvanced


class ImageData(object):
    """
    Stores the metadata for a Image Data groups.

    """

    def __init__(self, name, root_folder, image_folder=None, meta_folder=None,
                 image_data_generator=None):
        self._name = name
        self._root_folder = root_folder
        self._image_folder = self.path_setter(image_folder)
        self._meta_folder = self.path_setter(meta_folder)
        self._size = 0
        # self._image_data_generator = None
        self.image_data_generator = image_data_generator

    def path_setter(self, value):
        if value is None:
            return self.root_folder
        if os.path.isabs(value):
            path = value if os.path.exists(value) else self.root_folder
        else:
            tmp_path = os.path.join(self.root_folder, value)
            path = tmp_path if os.path.exists(tmp_path) else self.root_folder
        return path

    @property
    def name(self):
        return self._name

    @property
    def root_folder(self):
        return self._root_folder

    @root_folder.setter
    def root_folder(self, value):
        if os.path.exists(value):
            self._root_folder = value
        else:
            raise ValueError("Image Folder not found!")

    @property
    def image_folder(self):
        return self._image_folder

    @image_folder.setter
    def image_folder(self, value):
        self._image_folder = self.path_setter(value)

    @property
    def meta_folder(self):
        return self._meta_folder

    @meta_folder.setter
    def meta_foler(self, value):
        self._meta_folder = self.path_setter(value)

    @property
    def size(self):
        return self._size

    @property
    def image_data_generator(self):
        return self._image_data_generator

    @image_data_generator.setter
    def image_data_generator(self, value):
        if value is None:
            self._image_data_generator = ImageDataGenerator()
        elif isinstance(value, ImageDataGenerator):
            self._image_data_generator = value
        else:
            Warning("Not supported ")

    def to_string(self):
        return self.name

    def get_train(self, **kwargs):
        return self._get_train(**kwargs)

    def get_valid(self, **kwargs):
        return self._get_valid(**kwargs)

    def get_test(self, **kwargs):
        return self._get_test(**kwargs)

    @abstractmethod
    def _get_train(self, **kwargs):
        pass

    @abstractmethod
    def _get_valid(self, **kwargs):
        pass

    @abstractmethod
    def _get_test(self, **kwargs):
        pass


class ClassificationImageData(ImageData):
    """
    Version 1.
        Contains a ImageIterator for training, (potentially validation and testing)
        Add all related
    Version 2.

    """
    # TODO Version 2 support load nparray validation data for tensorboard to print histogram.

    def __init__(self, root_folder, image_dir=None, category=None, name=None, meta_folder=None,
                 use_validation=False,
                 train_image_gen_configs=None,
                 valid_image_gen_configs=None,
                 **kwargs):
        # self.root_folder = root_folder
        self.foo = 'foo'
        # self.image_folder = image_dir
        if name is None:
            name = os.path.split(root_folder)[1]
        super(ClassificationImageData, self).__init__(name=name, root_folder=root_folder,
                                                      image_folder=image_dir, meta_folder=meta_folder,
                                                      **kwargs)
        self.category_path = os.path.join(self.root_folder, category) \
            if category is not None and not os.path.isabs(category) else category
        self.category_dict = None
        self.nb_class = 0
        self.build_category_dict()

        # Store the lists by the mode.
        self.image_list = dict()
        self.label_list = dict()

        # Image format default value
        self.data_format = 'channels_last'
        self.target_size = (224, 224)
        self.batch_size = 32
        self.channel = 'rgb'

        self.use_validation = use_validation
        self.train_image_gen_configs = train_image_gen_configs \
            if train_image_gen_configs else ImageDataGeneratorAdvanced.get_default_train_config()
        self.valid_image_gen_configs = valid_image_gen_configs \
            if valid_image_gen_configs else ImageDataGeneratorAdvanced.get_default_valid_config()

    def build_category_dict(self):
        self.category_dict = self._build_category_dict()

    # Abstract Methods to be override by each dataset

    @abstractmethod
    def _build_category_dict(self):
        """
        Abstract method:
            To read meta-file, or scan the dataset, to build the dictionary for image category.
            And also, set the nb-class attributes.

        Returns
        -------
        category
        """
        pass

    @abstractmethod
    def decode(self, path):
        """
        support method for load image from txt.
        decode the image file from text to the classes.

        Parameters
        ----------
        path

        Returns
        -------

        """
        pass

    @abstractmethod
    def build_image_label_lists(self):
        """
        Build Image Label list for training, testing and validation.
        The built image list should contains the absolute path. (which can be accessed anywhere)

        Returns
        -------

        """
        pass

    def generator(self, mode, indice, **kwargs):
        mode = mode + str(indice)
        return self._generator(mode,
                               **kwargs)

    # Private methods
    def _generator(self, mode, batch_size=None, target_size=None, image_data_generator=None, **kwargs):
        """

        Parameters
        ----------
        mode
        batch_size
        target_size
        image_data_generator
        kwargs

        Returns
        -------

        """
        # Setting the default value if not provided.
        if image_data_generator is None:
            image_data_generator = self.image_data_generator
        if target_size is None:
            target_size = image_data_generator.target_size \
                if hasattr(image_data_generator, 'target_size') else self.target_size
        if batch_size is None:
            batch_size = self.batch_size

        file_list = self.image_list[mode]
        label_list = self.label_list[mode]
        if batch_size is None:
            batch_size = self.batch_size
        if target_size is None:
            target_size = self.target_size

        gen = ImageIterator(
            file_list, label_list, self.nb_class,
            image_data_generator=image_data_generator,
            batch_size=batch_size,
            target_size=target_size,
            category_dict=self.category_dict,
            data_format=self.data_format,
            **kwargs
        )
        return gen

    # Private method
    def _get_train(self, index=0, **kwargs):
        return self.generator('train', index, **kwargs)

    def _get_valid(self, index=0, **kwargs):
        return self.generator('valid', index, **kwargs)

    def _get_test(self, index=0, **kwargs):
        return self.generator('test', index, **kwargs)

    def _set_train(self, image_list, label_list, index=0):
        self.image_list['train' + str(index)] = image_list
        self.label_list['train' + str(index)] = label_list

    def _set_valid(self, image_list, label_list, index=0):
        self.image_list['valid' + str(index)] = image_list
        self.label_list['valid' + str(index)] = label_list

    def _set_test(self, image_list, label_list, index=0):
        self.image_list['test' + str(index)] = image_list
        self.label_list['test' + str(index)] = label_list

    def _load_image_location_from_txt(self, path):
        """
        Load the image list and maps to indices, as described.

        Parameters
        ----------
        path : str  path to txt partition lists.

        Returns
        -------

        """
        with open(path, 'r') as f:
            file_lists = f.read().splitlines()
        image_list = [self.image_folder + p if os.path.isabs(p) else os.path.join(self.image_folder, p)
                      for p in file_lists]
        label_list = [self.decode(p) for p in file_lists]
        return image_list, np.asanyarray(label_list)

    def _load_dict_from_txt(self, path, decode=None):
        """

        Parameters
        ----------
        path
        decode : func(str)  decode each line of txt into the name of it.

        Returns
        -------

        """
        if path is None:
            path = self.category_path
        with open(self.category_path, 'r') as f:
            cate_list = f.read().splitlines()
        if decode:
            cate_list = [decode(p) for p in cate_list]
        self.nb_class = len(cate_list)
        return dict(zip(cate_list, range(len(cate_list))))


