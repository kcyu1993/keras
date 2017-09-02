# -*- coding: utf-8 -*-

import glob

import numpy as np
import os
from os import path
from kyu.engine.utils.data_utils import ClassificationImageData
from scipy.io import loadmat

import keras.backend as K
from kyu.utils.image import ImageDataGeneratorAdvanced, ImageIterator

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
## Imagenet

IMAGENET_VALID_GROUNDTRUTH_FILE = 'ILSVRC2014_clsloc_validation_ground_truth.txt'
IMAGENET_VALID_BLACKLIST_FILE = 'ILSVRC2014_clsloc_validation_blacklist.txt'


class ImageNet_v2(ClassificationImageData):
    """
    Image Net Classification data.

    """


    def __init__(self, root_folder,
                 use_validation=False, image_dir='Data/CLS-LOC',
                 name='ImageNet', meta_folder='ImageSets/CLS-LOC'):
        self.meta_file = path.join(root_folder, meta_folder, 'meta_clsloc.mat')
        self.mat = self.load_mat()
        super(ImageNet_v2, self).__init__(root_folder=root_folder, image_dir=image_dir,
                                          name=name, meta_folder=meta_folder,
                                          use_validation=use_validation)
        # Construct the image label list
        self.build_image_label_lists()

    def load_mat(self):
        meta_path = self.meta_file
        mat = loadmat(meta_path)
        return mat

    def decode(self, path):
        """ decode the image path info to string """
        raise NotImplementedError

    def build_image_label_lists(self):
        """
        Construct the ImageNet

        Returns
        -------

        """
        images = self.mat['images']

        train_name = images['name'][0, 0][np.where(images['set'][0, 0] == 1)]
        val_name = images['name'][0, 0][np.where(images['set'][0, 0] == 2)]
        if not self.use_validation:
            train_name = np.append(train_name, val_name)
        test_name = images['name'][0, 0][np.where(images['set'][0, 0] == 3)]

        y_train = images['class'][0, 0][np.where(images['set'][0, 0] == 1)]
        y_valid = images['class'][0, 0][np.where(images['set'][0, 0] == 2)]
        if not self.use_validation:
            y_train = np.append(y_train, y_valid)
        y_test = images['class'][0, 0][np.where(images['set'][0, 0] == 3)]

        y_train -= 1
        y_valid -= 1
        y_test -= 1

        self._set_train(
            [path.join(self.image_folder, file_name[0]) for file_name in train_name],
            y_train,
        )

        if self.use_validation:
            self._set_valid(
                [path.join(self.image_folder, file_name[0]) for file_name in val_name],
                y_valid,
            )

        self._set_test(
            [path.join(self.image_folder, file_name[0]) for file_name in test_name],
            y_test,
        )

    def _build_category_dict(self):
        """ Build the category dictionary by meta-file """
        meta = self.mat['meta']
        classes = meta['classes'][0][0][0]
        # To python str for unified usage
        classes_names = [str(classes[i][0]) for i in range(len(classes))]
        category_dict = dict(zip(classes_names, range(len(classes_names))))
        self.nb_class = len(classes_names)
        return category_dict


class ImageNetTools(object):
    """
    Define the ImageNet tools, synsets reading and loading.
    """
    def __init__(self, fpath):
        meta_clsloc_file = fpath
        self.synsets = loadmat(meta_clsloc_file)["synsets"][0]
        self.synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0]))
                                               for s in self.synsets[:1000]],
                                         key=lambda v: v[1])

        self.corr = {}
        for j in range(1000):
            self.corr[self.synsets_imagenet_sorted[j][0]] = j

        self.corr_inv = {}
        for j in range(1, 1001):
            self.corr_inv[self.corr[j]] = j

    def synset_to_id(self, synset):
        a = next((s[0] for s in self.synsets if s[1] == synset), None)
        # a = next((i for (i, s) in self.synsets_imagenet_sorted if s == synset), None)
        return int(a)

    def id_to_synset(self, id_):
        # return str(self.synsets[self.corr_inv[id_] - 1][1][0])
        return str(self.synsets[id_ - 1][1][0])

    def id_to_words(self, id_):
        # return self.synsets[self.corr_inv[id_] - 1][2][0]
        return self.synsets[id_ - 1][2][0]

    def nn_id_to_id(self, id_):
        """
        Sorted id into Real object id
        Parameters
        ----------
        id_: neural net id

        Returns
        -------
        Real Synset corresponding ID
        """
        return self.corr_inv[int(id_)]

    def id_to_nnid(self, id_):
        return self.corr[int(id_)]

    def synset_to_nn_id(self, synset):
        a = next((s[0] for s in self.synsets if s[1] == synset), None)
        # a = next((i for (i, s) in self.synsets_imagenet_sorted if s == synset), None)
        return self.corr[int(a)]


class ImageNetLoader(object):
    """
    ImageNet Loader
    Create for Imagenet loading process

    Directory structure:
    path/to/imagenet/root/

        ImageSets/
            CLS-LOC/
            # Contains the pre-defined train, valid and test images paths.
            # cate_id varies from 1 to 1000
                train_cls.txt
                    nXXXXXX/nXXXXX_XXX.JPEG cate_id     # ID/ID_subID.JPEG category id
                    ...

                val.txt
                test.txt

        Data/               Contains the images
            CLS-LOC/
                train/
                val/
                test/

        Annotations/
            # NOT FOR NOW



    """
    def __init__(self, dirpath='/home/kyu/.keras/datasets/ILSVRC2015', metadata_path=None, mode='CLS-LOC',
                 data_folder='Data', info_folder='ImageSets',
                 train='train', valid='val', test='test',
                 config_fname='imagenet_keras_pipeline_list.h5',
                 dim_ordering='default'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering

        self.nb_class = 1000

        self.directory = dirpath
        # Make path to be absolute
        self.mode = mode
        self.data_folder = os.path.join(dirpath, data_folder, self.mode)
        self.info_folder = os.path.join(dirpath, info_folder, self.mode)

        self.train_name = train
        self.valid_name = valid
        self.test_name = test

        # Get class_number list.
        tmp_classes = glob.glob(os.path.join(self.data_folder, train, '*'))
        self.classes = [os.path.split(l)[1] for l in tmp_classes]

        # Should load from
        if metadata_path is None:
            metadata_path = os.path.join(self.info_folder, 'meta_clsloc.mat')
        self.metadata_path = metadata_path
        # Create the ImageNet tools
        self.imagenettool = ImageNetTools(self.metadata_path)

        # Define key-value pair for WnID and indices
        self.classes_indices = {k:v for k,v in
                                zip(self.classes,
                                    [self.imagenettool.synset_to_id(n)
                                     for n in self.classes])
                                }

        # Load Valid txt file.
        valid_groundtruth_txt = open(os.path.join(self.info_folder, IMAGENET_VALID_GROUNDTRUTH_FILE))
        valid_groundtruth_list = [l for l in valid_groundtruth_txt.readlines()]
        self.valid_groundtruth_dict = dict(zip(range(len(valid_groundtruth_list)), valid_groundtruth_list))

        ######################
        # Assert the loading logic
        ######################

        load_txt = False
        load_train, load_valid, load_test = False, False, False

        try:
            import h5py
            fdata = h5py.File(os.path.join(self.info_folder, config_fname), 'r')
            try:
                self.train_list = fdata['train_list']
            except KeyError as e:
                print(str(e))
                load_train = True
            try:
                self.valid_list = fdata['valid_list']
            except KeyError as e:
                print(str(e))
                load_valid = True
            try:
                self.test_list = fdata['test_list']
            except KeyError as e:
                print(str(e))
                load_test = True
        except (ImportError, IOError) as e:
            print(str(e))
            load_txt = True

        if load_txt:
            print("Loading images list from given txt file ")
            load_train = True
            load_valid = True
            load_test = True

        if load_train:
            train_txt = open(os.path.join(self.info_folder, 'train_cls.txt'), 'r')
            print("from train_cls.txt ...")
            train_all = train_txt.readlines()

            """ Small sample settings"""
            # shuffle(train_all)
            # train_all = train_all[:100000]
            self.train_list = np.asanyarray([self.decode(l, mode='train', prefix='train') for l in train_all])
            # self.train_list = np.asanyarray([self.decode(l, mode='train') for l in train_all[:10000]])
        if load_valid:
            valid_txt = open(os.path.join(self.info_folder, 'val_cls.txt'), 'r')
            print("from valid.txt ...")
            self.valid_list = np.asanyarray([self.decode(l, mode='train', prefix='val') for l in valid_txt.readlines()])
        if load_test:
            test_txt = open(os.path.join(self.info_folder, 'test.txt'), 'r')
            print('from test.txt ...')
            self.test_list = np.asanyarray([self.decode(l, mode='test') for l in test_txt.readlines()])

    def generator(self, mode='valid', batch_size=32, target_size=(224,224),
                  image_data_generator=None, **kwargs):
        if not mode in {'train', 'valid', 'test'}:
            raise ValueError('Mode should be one of train, valid, and test')
        if mode == 'train':
            flist = self.train_list[:, 1]
            cate_list = self.train_list[:, 3]
        elif mode == 'valid':
            flist = self.valid_list[:, 1]
            cate_list = self.valid_list[:, 3]
        elif mode == 'test':
            flist = self.test_list[:, 1]
            cate_list = self.test_list[:, 3]
        else:
            raise ValueError
        # Transform the list type.
        # if isinstance(flist, np.ndarray):
        #     flist = flist.tolist()
        # if isinstance(cate_list, np.ndarray):
        #     cate_list = cate_list.tolist()
        generator = ImageIterator(flist, cate_list, self.nb_class,
                                  image_data_generator=image_data_generator,
                                  batch_size=batch_size, target_size=target_size,
                                  dim_ordering=self.dim_ordering, **kwargs)
        return generator

    def decode(self, str_input, mode='train', prefix=None):
        """
        Decode the corresponding parameters

        Parameters
        ----------
        str_input: str  one line of given txt file
                        for training:
                            WnID/WnID_imgID.JPEG Index
                        for validation and test
                            XXXX_XXXX_ID.JPEG Index
        Returns
        -------
        index, abs_path, corresponding SynsetID (-1 for test)
        """
        path, index = str_input.split(' ')

        if mode == 'train':
            synset, _ = path.split('/')
            synset_id = self.imagenettool.synset_to_id(synset)
            name = self.train_name
        elif mode == 'valid':
            synset_id = self.valid_groundtruth_dict[int(index) - 1]
            name = self.valid_name
        elif mode == 'test':
            synset_id = -1
            name = self.test_name
        else:
            raise ValueError("Only accept test, valid, train as mode")

        # if not os.path.exists(abs_path):
        #     raise IOError('File not found ' + abs_path)
        if prefix:
            name = prefix
        abs_path = os.path.join(self.data_folder, name, path + '.JPEG')
        if synset_id == -1:
            nnid = 0
        else:
            nnid = self.imagenettool.id_to_nnid(synset_id)
        return int(index), abs_path, int(synset_id), nnid


def save_list_to_h5df():
    import h5py

    IMAGENET_PATH = '/home/kyu/.keras/datasets/ILSVRC2015'
    TARGET_SIZE = (224, 224)
    RESCALE_SMALL = 256

    BATCH_SIZE = 16
    NB_EPOCH = 70
    VERBOSE = 1
    SAVE_LOG = False

    # ImageNet generator
    imageNetLoader = ImageNetLoader(IMAGENET_PATH)
    gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                     horizontal_flip=True,
                                     channelwise_std_normalization=True)
    SAVE_PATH = IMAGENET_PATH + '/ImageSets/CLS-LOC/'
    print("Saving the file to " + SAVE_PATH)
    f = h5py.File(SAVE_PATH + 'imagenet_keras_pipeline_lists.h5', 'w')
    f.create_dataset('train_list', data=imageNetLoader.train_list,
                     compression='gzip', compression_opts=9)
    f.create_dataset('valid_list', data=imageNetLoader.valid_list,
                     compression='gzip', compression_opts=9)
    f.create_dataset('test_list', data=imageNetLoader.test_list,
                     compression='gzip', compression_opts=9)

if __name__ == '__main__':
    save_list_to_h5df()
