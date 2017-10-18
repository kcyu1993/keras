"""
SUN 397 standard dataset

"""
import os
import numpy as np

from kyu.utils.dict_utils import create_dict_by_given_kwargs
from kyu.utils.image import ImageIterator
from kyu.engine.utils.data_utils import ClassificationImageData


class SUN397(object):
    """
    SUN dataset includes the following dataset structure

        a/
            axxxxx
            abbbbb
            ...
        b/
            ...
        ...


    With partitions storing in
        Testing_01.txt
        Training_01.txt
        Valid_01.txt


    """

    def __init__(self, dirpath='', category='ClassName.txt', image_dir='SUN397'):
        self.r_path = dirpath # should be .../sun/
        self.category_path = os.path.join(dirpath, category)
        self.image_path = os.path.join(dirpath, image_dir)

        # Load the category
        with open(self.category_path, 'r') as f:
            cate_list = f.read().splitlines()

        # Remove the first
        cate_list = [p.split('/')[2] for p in cate_list]

        self.category_dict = dict(zip(cate_list, range(len(cate_list))))
        self.nb_class = len(cate_list)
        self.image_list = None
        self.label_list = None

        self.dim_ordering = 'tf'

    def decode(self, path):
        """
        Relative path, give the resulting category

        Parameters
        ----------
        path

        Returns
        -------

        """

        return self.category_dict[path.split('/')[2]]

    def load_image_location_from_txt(self, path):
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

        self.image_list = [self.image_path + p for p in file_lists]
        self.label_list = [self.decode(p) for p in file_lists]
        return self.image_list, np.asanyarray(self.label_list)

    def generator(self, mode='train', indice=1, batch_size=32,
                  target_size=(224, 224), image_data_generator=None, **kwargs):
        """
        Get the mode and indices
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
            txt_file = 'Training_{:02d}.txt'.format(indice)
        elif mode == 'test':
            txt_file = 'Testing_{:02d}.txt'.format(indice)
        else:
            raise ValueError("mode not supported {}".format(mode))

        flist, cate_list = self.load_image_location_from_txt(os.path.join(self.r_path, txt_file))

        generator = ImageIterator(
            flist, cate_list, self.nb_class,
            image_data_generator=image_data_generator,
            batch_size=batch_size,
            target_size=target_size,
            category_dict=self.category_dict,
            dim_ordering=self.dim_ordering,
            **kwargs
        )
        return generator


class SUN397_v2(ClassificationImageData):

    def __init__(self, dirpath='', category='ClassName.txt', image_dir='SUN397'):
        super(SUN397_v2, self).__init__(dirpath, image_dir, category, name='SUN397')
        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[256, 296], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=256, random_crop=False, horizontal_flip=True)

    def build_image_label_lists(self):
        for i in range(1,11):
            train_file = 'Training_{:02d}.txt'.format(i)
            test_file = 'Testing_{:02d}.txt'.format(i)
            train_img, train_label = self._load_image_location_from_txt(os.path.join(self.root_folder, train_file))
            test_img, test_label = self._load_image_location_from_txt(os.path.join(self.root_folder, test_file))
            self._set_train(train_img, train_label, index=i - 1)
            self._set_test(test_img, test_label, index=i - 1)

    def _build_category_dict(self):
        # # Load the category
        with open(self.category_path, 'r') as f:
            cate_list = f.read().splitlines()
        # Remove the first
        cate_list = [p.split('/')[2] for p in cate_list]
        self.nb_class = len(cate_list)
        return dict(zip(cate_list, range(len(cate_list))))

    def decode(self, path):
        return self.category_dict[path.split('/')[2]]


if __name__ == '__main__':
    # PASS the test.
    sun = SUN397('/home/kyu/.keras/datasets/sun')
    gen = sun.generator(save_to_dir='/home/kyu/cvkyu/plots', save_prefix='sundataset')
    a, b = gen.next()


