"""
Create the reader for Amazon.shoes category.

Version 0.1
    Support the creation and threshold for label

"""
import random

import os

from kyu.utils.dict_utils import create_dict_by_given_kwargs

from kyu.utils.data_utils import csv_reader, csv_saver

from kyu.engine.utils.data_utils import ClassificationImageData
from kyu.utils.io_utils import save_list_to_file, load_list_from_file
from keras.preprocessing.image import load_img

def clean_csv_filelist(file_lists, root_dir='/home/kyu/.keras/datasets/amazon'):
    """
    Clean the CSV Filelist

    Parameters
    ----------
    entries
    root_dir

    Returns
    -------

    """
    remove_index = []
    for ind, e in enumerate(file_lists):
        e = e[1:]
        img_path = os.path.join(root_dir, e)
        try:
            img = load_img(img_path)
        except IOError:
            remove_index.append(ind)

    return remove_index


def clean_csv(csv_file):
    root_dir = '/home/kyu/.keras/datasets/amazon'
    csv_file = '/home/kyu/.keras/datasets/amazon/20171214-Shoes_smallrank.csv'

    title, entries = csv_reader(csv_file)
    filelists = [e[0] for e in entries]
    remove_indices = clean_csv_filelist(filelists, root_dir)
    # remove indices
    remove_indices.reverse()
    for i in remove_indices:
        del entries[i]
    print("Cleaned results {}".format(len(entries)))
    revised_csv_file = '/home/kyu/.keras/datasets/amazon/20171214-Shoes_smallrank-cleaned.csv'
    csv_saver(revised_csv_file, entries, title)


class AmazonBestSellerSmallrank(ClassificationImageData):

    def __init__(self, dir="/home/kyu/.keras/datasets/amazon",
                 threshold=30,
                 meta_file='20171214-Shoes_smallrank-cleaned.csv',
                 generate_random_indices=True,
                 **kwargs):
        """
        Create Best seller
            Item rank at the Threshold, is treated as Best-seller (with label 1)

        Parameters
        ----------
        dir
        """
        self.generate_random_indices = generate_random_indices
        self.threshold = threshold  # to cut the data
        super(AmazonBestSellerSmallrank, self).__init__(dir, **kwargs)
        self.csv_file = os.path.join(self.root_folder, meta_file)
        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[256, 296], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=296, random_crop=False, horizontal_flip=True)


    def build_category_dict(self):
        # return None
        category_dict = {}
        self.nb_class = 2
        for i in range(1, 101):
            category_dict[str(i)] = 0 if i > self.threshold else 1
        self.category_dict = category_dict
        return category_dict

    def build_image_label_lists(self):
        """

        Returns
        -------

        """
        # load the image path and the image rank
        title, entries = csv_reader(self.csv_file)
        # random split by 0.8 vs 0.2, for train and test, no valid for the moment
        nb_sample = len(entries)
        if self.generate_random_indices:
            rand_indices = range(0, len(entries))
            random.shuffle(rand_indices)
            save_list_to_file(rand_indices, os.path.join(self.root_folder, "current_index.json"))
        else:
            rand_indices = load_list_from_file(os.path.join(self.root_folder, "current_index.json"))
        cutting = int(nb_sample * 0.8)

        train_image = [self.absolute_image_path(entries[i][0]) for i in rand_indices[0:cutting]]
        train_label = [self.category_dict[entries[i][1]] for i in rand_indices[0:cutting]]
        test_image = [self.absolute_image_path(entries[i][0]) for i in rand_indices[cutting:]]
        test_label = [self.category_dict[entries[i][1]] for i in rand_indices[cutting:]]

        self._set_train(train_image, train_label)
        self._set_test(test_image, test_label)

    def absolute_image_path(self, path):
        """
        Remove the first
        Parameters
        ----------
        path

        Returns
        -------

        """
        if os.path.isabs(path):
            path = path[1:]
        return os.path.join(self.root_folder, path)


if __name__ == '__main__':
    clean_csv("")