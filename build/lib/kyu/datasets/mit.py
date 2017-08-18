"""
MIT indoor scene dataset

"""
import os
import numpy as np

from kyu.datasets.imagenet import ImageIterator


class MITLoader(object):
    """
    MIT dataset loader
    Created for loading MIT indoor scene recognition problem
    
    
    """

    def __init__(self, dirpath='', category='category.txt', image_dir='Images'):
        self.r_path = dirpath
        self.category_path = os.path.join(dirpath, category)
        self.image_path = os.path.join(dirpath, image_dir)

        # Load the category
        with open(self.category_path, 'r') as f:
            cate_list = f.read().splitlines()
        self.category_dict = dict(zip(cate_list, range(len(cate_list))))
        self.nb_class = len(cate_list)
        self.image_lists = None
        self.label_lists = None
        self.dim_ordering = 'tf'

    def decode(self, path):
        """
        Give a relative path, return the results category
        Parameters
        ----------
        path

        Returns
        -------

        """
        return self.category_dict[path.split('/')[0]]

    def load_image_location_from_txt(self, path):
        """
        Load the image as list and maps to a indices.

        Parameters
        ----------
        path

        Returns
        -------

        """
        with open(path, 'r') as f:
            file_lists = f.read().splitlines()
        self.image_lists = [os.path.join(self.image_path, p) for p in file_lists]
        self.label_lists = [self.decode(p) for p in file_lists]

        return self.image_lists, np.asanyarray(self.label_lists)

    def generator(self, mode='train', batch_size=32, target_size=(224,224), image_data_generator=None, **kwargs):
        """
        get the mode.
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
        if mode == 'train':
            txt_file = 'TrainImages.txt'
        elif mode == 'complete':
            # txt_file = 'NewTrain.txt'
            txt_file = 'NewTrain.txt'
        elif mode == 'complete_train':
            txt_file = 'CompleteTrain.txt'
        else:
            txt_file = 'TestImages.txt'

        flist, cate_list = self.load_image_location_from_txt(os.path.join(self.r_path, txt_file))

        generator = ImageIterator(flist, cate_list, self.nb_class,
                                  image_data_generator=image_data_generator,
                                  batch_size=batch_size, target_size=target_size,
                                  dim_ordering=self.dim_ordering, **kwargs
                                  )
        return generator

