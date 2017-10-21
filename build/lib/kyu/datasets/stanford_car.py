"""
Stanford Car reader

"""
from kyu.utils.image import ImageDataGeneratorAdvanced

from kyu.utils.dict_utils import create_dict_by_given_kwargs

from kyu.engine.utils.data_utils import ClassificationImageData
from os import path
from scipy.io import loadmat
import numpy as np


class StanfordCar(ClassificationImageData):

    def __init__(self, root_folder, use_validation=False, image_dir='', name='StanfordCar', meta_folder=''):
        self.meta_file = path.join(root_folder, meta_folder, 'cars_annos.mat')
        self.mat = self.load_mat()
        super(StanfordCar, self).__init__(root_folder=root_folder, image_dir=image_dir, name=name, meta_folder=meta_folder,
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
        images = self.mat['annotations']

        train_name = images['relative_im_path'][0][np.where(images['test'][0] == 0)]
        test_name = images['relative_im_path'][0][np.where(images['test'][0] == 1)]
        train_name = np.asanyarray([name[0] for name in train_name])
        test_name = np.asanyarray([name[0] for name in test_name])
        y_train = images['class'][0][np.where(images['test'][0] == 0)] - 1
        y_test = images['class'][0][np.where(images['test'][0] == 1)] - 1
        y_train = np.asanyarray([y[0][0] for y in y_train])
        y_test = np.asanyarray([y[0][0] for y in y_test])

        self._set_train(
            [path.join(self.image_folder, file_name) for file_name in train_name],
            y_train,
        )

        self._set_test(
            [path.join(self.image_folder, file_name) for file_name in test_name],
            y_test,
        )

    def _build_category_dict(self):
        """ Build the category dictionary by meta-file """
        classes = self.mat['class_names'][0]
        # To python str for unified usage
        classes_names = [str(classes[i][0]) for i in range(len(classes))]
        category_dict = dict(zip(classes_names, range(len(classes_names))))
        self.nb_class = len(classes_names)
        return category_dict

if __name__ == '__main__':
    data = StanfordCar('/home/kyu/.keras/datasets/car')
    TARGET_SIZE = (448, 448)
    RESCALE_SMALL = [2 * 256, 820]
    gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                     horizontal_flip=True,
                                     )
    valid_gen = ImageDataGeneratorAdvanced(TARGET_SIZE,
                                           rescaleshortedgeto=448,
                                           random_crop=False,
                                           horizontal_flip=True)
    # def label_wrapper(label):
    #     return data.category_dict[label]

    valid = data.get_test(image_data_generator=valid_gen, save_to_dir='/home/kyu/plots',
                          save_prefix='car_valid', save_format='JPEG',
                          shuffle=False)

    train = data.get_train(image_data_generator=gen, save_to_dir='/home/kyu/plots',
                           save_prefix='car', save_format='JPEG',
                           shuffle=False)

    a, b = valid.next()
    a, b = train.next()
