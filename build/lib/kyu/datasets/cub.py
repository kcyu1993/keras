"""
Build CUB dataset loader

"""
from ..datasets.common_imports import *


class CUB(ClassificationImageData):
    """
    CUB dataset accordingly
    """

    def __init__(self, dirpath='', category='classes.txt', image_dir="images",
                 ):
        super(CUB, self).__init__(dirpath, image_dir, category, name='CUB2011')
        self.image_list_file = 'images.txt'
        self.image_label_file = 'image_class_labels.txt'
        self.split_file = 'train_test_split.txt'
        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=449, random_crop=False, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=449, random_crop=False, horizontal_flip=True)

    def build_image_label_lists(self):
        # read the images.txt
        with open(os.path.join(self.root_folder, self.image_list_file)) as f:
            file_lists = f.read().splitlines()
        with open(os.path.join(self.root_folder, self.split_file)) as f:
            split_lists = f.read().splitlines()
        with open(os.path.join(self.root_folder, self.image_label_file)) as f:
            label_verify_lists = f.read().splitlines()

        file_lists = [os.path.join(self.image_folder, self.decode(p)[1]) for p in file_lists]
        split_lists = [int(self.decode(p)[1]) for p in split_lists]
        label_verify_lists = [int(self.decode(p)[1]) - 1 for p in label_verify_lists]

        train_ind = np.where(np.asanyarray(split_lists) == 1)
        test_ind = np.where(np.asanyarray(split_lists) == 0)
        file_lists = np.asanyarray(file_lists)
        label_verify_lists = np.asanyarray(label_verify_lists)
        self._set_train(file_lists[train_ind].tolist(), label_verify_lists[train_ind])
        self._set_test(file_lists[test_ind].tolist(), label_verify_lists[test_ind])

    def _build_category_dict(self):
        def _decode(line):
            return str(line).split('.')[1]
        return self._load_dict_from_txt(self.category_path, _decode)

    def decode(self, path):
        """ return the id and info """
        return str(path).split(' ')


if __name__ == '__main__':
    data = CUB('/home/kyu/.keras/datasets/cub/CUB_200_2011')
    TARGET_SIZE = (448, 448)
    RESCALE_SMALL = [2*256, 592]
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
                          save_prefix='cub_valid', save_format='JPEG',
                          shuffle=False)

    train = data.get_train(image_data_generator=gen, save_to_dir='/home/kyu/plots',
                           save_prefix='cub', save_format='JPEG',
                           shuffle=False)

    a, b = valid.next()
    a, b = train.next()
