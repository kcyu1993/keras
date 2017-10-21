"""
Airplane dataset.

"""

from kyu.datasets.common_imports import *


class Aircraft(ClassificationImageData):
    """
    Aircraft dataset

    """

    def __init__(self, dirpath, level='variant', image_dir='images'):
        super(Aircraft, self).__init__(
            dirpath, image_dir, level + 's.txt', name='AirCraft_' + level)
        self.level = level
        self.category_dict = self._build_category_dict()
        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[225, 296], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=256, random_crop=False, horizontal_flip=True)

    def build_image_label_lists(self):
        # Read the images
        file_list = ['train', 'val', 'test']
        file_list = [os.path.join(self.root_folder, 'images_{}_{}.txt'.format(self.level, p)) for p in file_list]
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
        self._set_train(img_list[0] + img_list[1], np.asanyarray(label_list[0] + label_list[1]))
        # self._set_valid(img_list[1], np.asanyarray(label_list[1]))
        self._set_test(img_list[2], np.asanyarray(label_list[2]))

    def decode(self, path):
        ind = str(path).find(' ')
        return str(path)[:ind] + '.jpg', self.category_dict[str(path)[ind+1:]]

    def _build_category_dict(self):
        return self._load_dict_from_txt(self.category_path)

if __name__ == '__main__':
    data = Aircraft('/home/kyu/.keras/datasets/fgvc-aircraft-2013b/data')
    TARGET_SIZE = (448, 448)
    RESCALE_SMALL = [2 * 256, 592]
    gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                     horizontal_flip=True,
                                     )
    valid_gen = ImageDataGeneratorAdvanced(TARGET_SIZE,
                                           rescaleshortedgeto=512,
                                           random_crop=False,
                                           horizontal_flip=True)
    # def label_wrapper(label):
    #     return data.category_dict[label]

    valid = data.get_test(image_data_generator=valid_gen, save_to_dir='/home/kyu/plots',
                          save_prefix='aircraft_valid', save_format='JPEG',
                          shuffle=False)

    train = data.get_train(image_data_generator=gen, save_to_dir='/home/kyu/plots',
                           save_prefix='aircraft', save_format='JPEG',
                           shuffle=False)

    a, b = valid.next()
    a, b = train.next()