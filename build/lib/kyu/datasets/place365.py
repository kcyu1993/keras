"""
Get the place 365 datasets

"""
from kyu.engine.utils.data_utils import ClassificationImageData
import os
import numpy as np
import ast
from kyu.utils.dict_utils import create_dict_by_given_kwargs


class Place256(ClassificationImageData):
    """
    Directory structure:
    places_devkit/
        places_category.txt
        places365_standard_train.txt
        places365_validation.txt
        places365_test.txt

    """
    def __init__(self, dirpath='', category='categories_places365.txt',
                 meta_folder='places_devkit',
                 image_dir='data_256',
                 val_folder='val_256',
                 label_dir='places_devkit',
                 **kwargs):
        super(Place256, self).__init__(root_folder=dirpath, image_dir=image_dir,
                                       category=category,
                                       meta_folder=meta_folder,
                                       name='place256',
                                       **kwargs)

        self.label_dir = os.path.join(self.root_folder, label_dir)
        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[256, 296], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=296, random_crop=False, horizontal_flip=True)
        self.val_folder = os.path.join(self.root_folder, val_folder)

    def build_category_dict(self):
        pass

    def build_image_label_lists(self):
        # Build the image label list based on the txt file
        train_file = 'places365_train_standard.txt'
        test_file = 'places365_val.txt'
        train_img, train_label = self._load_image_location_from_txt(os.path.join(self.label_dir, train_file))
        test_img, test_label = self._load_image_location_from_txt(os.path.join(self.label_dir, test_file))

        self._set_train(train_img, train_label, index=0)
        self._set_test(test_img, test_label, index=0)

    def decode(self, path):
        pass

    def _load_image_location_from_txt(self, path):
        """
        Load the image list and maps to indices, as described.

        Parameters
        ----------
        path : str  path to txt partition lists.

        Returns
        -------

        """
        if 'val' in path:
            image_folder = self.val_folder
        else:
            image_folder = self.image_folder

        with open(path, 'r') as f:
            file_lists = f.read().splitlines()
        image_list = []
        label_list = []
        for p in file_lists:
            img, label = p.split(' ')
            image_list.append(image_folder + img if os.path.isabs(img) else os.path.join(image_folder, img))
            label_list.append(ast.literal_eval(label))
        return image_list, np.asanyarray(label_list)
