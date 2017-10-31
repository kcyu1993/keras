"""
Define the food 101 dataset

"""
import os
import numpy as np
from kyu.utils.image import ImageDataGeneratorAdvanced

from kyu.utils.dict_utils import create_dict_by_given_kwargs

from kyu.engine.utils.data_utils import ClassificationImageData


class Food101(ClassificationImageData):

    def __init__(self, root_folder, image_dir='images',
                 meta_folder='meta', category='classes.txt'):
        super(Food101, self).__init__(root_folder, image_dir,
                                      meta_folder=meta_folder,
                                      category=category, name='Food-101')
        # build the index
        self.category_path = os.path.join(self.meta_folder, category)
        self.category_dict = self._load_dict_from_txt(self.category_path)

        self.build_image_label_lists()

        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[225, 296], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=225, random_crop=False, horizontal_flip=True)

    def build_image_label_lists(self):
        # Read the train and txt file
        file_list = ['train', 'test']
        file_list = [os.path.join(self.meta_folder, '{}.txt'.format(p)) for p in file_list]
        img_list = []
        label_list = []
        for ind, l in enumerate(file_list):
            img_list.append([])
            label_list.append([])
            with open(l, 'r') as f:
                lines = f.read().splitlines()

                for p in lines:
                    file_path, label = self.decode(p)
                    img_list[ind].append(os.path.join(self.image_folder, file_path))
                    label_list[ind].append(label)

        # combine the train and val
        self._set_train(img_list[0], np.asanyarray(label_list[0]))
        # self._set_valid(img_list[1], np.asanyarray(label_list[1]))
        self._set_test(img_list[1], np.asanyarray(label_list[1]))

    def decode(self, path):
        label = self.category_dict[path.split('/')[0]]
        return path + '.jpg', label

if __name__ == '__main__':
    data = Food101('/home/kyu/.keras/datasets/food-101/food-101')
    TARGET_SIZE = (224, 224)
    RESCALE_SMALL = [256, 296]
    gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                     horizontal_flip=True,
                                     )
    valid_gen = ImageDataGeneratorAdvanced(TARGET_SIZE,
                                           rescaleshortedgeto=256,
                                           random_crop=False,
                                           horizontal_flip=True)
    # def label_wrapper(label):
    #     return data.category_dict[label]

    valid = data.get_test(image_data_generator=valid_gen, save_to_dir='/home/kyu/plots',
                          save_prefix='food_valid', save_format='JPEG',
                          shuffle=False)

    train = data.get_train(image_data_generator=gen, save_to_dir='/home/kyu/plots',
                           save_prefix='food', save_format='JPEG',
                           shuffle=False)

    a, b = valid.next()
    a, b = train.next()
