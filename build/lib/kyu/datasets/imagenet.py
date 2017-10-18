# -*- coding: utf-8 -*-

import glob

import numpy as np
import os
from os import path
from kyu.engine.utils.data_utils import ClassificationImageData
from scipy.io import loadmat

import keras.backend as K
from kyu.utils.dict_utils import create_dict_by_given_kwargs
from kyu.utils.image import ImageDataGeneratorAdvanced, ImageIterator

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
## Imagenet

IMAGENET_VALID_GROUNDTRUTH_FILE = 'ILSVRC2014_clsloc_validation_ground_truth.txt'
IMAGENET_VALID_BLACKLIST_FILE = 'ILSVRC2014_clsloc_validation_blacklist.txt'


class ImageNetData(ClassificationImageData):
    """
    Image Net Classification data.

    """
    def __init__(self, root_folder,
                 use_validation=False, image_dir='Data',
                 name='ImageNet', meta_folder='ImageSets',
                 mode='CLS-LOC',
                 config_filename='imagenet_keras_pipeline_list.h5'):
        self.meta_file = path.join(root_folder, meta_folder, mode, 'meta_clsloc.mat')
        self.config_file = path.join(root_folder, meta_folder, mode, config_filename)
        self.imagenettool = ImageNetTools(self.meta_file)
        self.train_name = 'train'
        self.valid_name = 'val'
        self.test_name = 'test'
        self.mat = self.load_mat()
        super(ImageNetData, self).__init__(root_folder=root_folder, image_dir=os.path.join(image_dir, mode),
                                           name=name, meta_folder=os.path.join(meta_folder, mode),
                                           use_validation=use_validation)
        # Construct the image label list

        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[256, 512], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=256, random_crop=False, horizontal_flip=True)

    def load_mat(self):
        meta_path = self.meta_file
        mat = loadmat(meta_path)
        return mat

    def decode(self, path):
        """
        decode the image path info to string
        For training
            WnID/WnID_imgID.jpeg
        For validation
            XXXX_XXXX_ID.jpeg
        For testing
            No idea the real value

        Parameters
        ----------
        path

        Returns
        -------

        """
        raise NotImplementedError

    def imagenet_decode(self, str_input, mode='train', prefix=None):
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
        abs_path = os.path.join(self.image_folder, name, path + '.JPEG')
        if synset_id == -1:
            nnid = 0
        else:
            nnid = self.imagenettool.id_to_nnid(synset_id)
        return int(index), abs_path, int(synset_id), nnid

    def build_image_label_lists(self):
        """
        Construct the ImageNet

        Returns
        -------

        """
        load_txt = False
        load_valid = False
        load_train = False
        load_test = False
        try:
            import h5py
            fdata = h5py.File(os.path.join(self.meta_folder, self.config_file), 'r')
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
            train_txt = open(os.path.join(self.meta_folder, 'train_cls.txt'), 'r')
            print("from train_cls.txt ...")
            train_all = train_txt.readlines()

            """ Small sample settings"""
            # shuffle(train_all)
            # train_all = train_all[:100000]
            self.train_list = np.asanyarray([self.imagenet_decode(l, mode='train', prefix='train') for l in train_all])
            # self.train_list = np.asanyarray([self.decode(l, mode='train') for l in train_all[:10000]])
        if load_valid:
            valid_txt = open(os.path.join(self.meta_folder, 'val_cls.txt'), 'r')
            print("from valid.txt ...")
            self.valid_list = np.asanyarray([self.imagenet_decode(l, mode='train', prefix='val') for l in valid_txt.readlines()])
        if load_test:
            test_txt = open(os.path.join(self.meta_folder, 'test.txt'), 'r')
            print('from test.txt ...')
            self.test_list = np.asanyarray([self.imagenet_decode(l, mode='test') for l in test_txt.readlines()])

        self._set_train(
            self.train_list[:, 1],
            self.train_list[:, 3]
        )
        if self.use_validation:
            self._set_valid(
                self.valid_list[:, 1],
                self.valid_list[:, 3]
            )
            self._set_test(
                self.test_list[:, 1],
                self.test_list[:, 3]
            )
        else:
            self._set_test(
                self.valid_list[:, 1],
                self.valid_list[:, 3]
            )

    def _build_category_dict(self):
        """ Build the category dictionary by meta-file """
        tmp_classes = glob.glob(os.path.join(self.image_folder, self.train_name, '*'))
        classes_names = [os.path.split(l)[1] for l in tmp_classes]

        # To python str for unified usage
        category_dict = {k:v for k,v in
                         zip(classes_names,
                             [self.imagenettool.synset_to_nn_id(n)
                              for n in classes_names])
                         }
        self.nb_class = len(classes_names)
        assert self.nb_class == 1000
        valid_groundtruth_txt = open(os.path.join(self.meta_folder, IMAGENET_VALID_GROUNDTRUTH_FILE))
        valid_groundtruth_list = [l for l in valid_groundtruth_txt.readlines()]
        self.valid_groundtruth_dict = dict(zip(range(len(valid_groundtruth_list)), valid_groundtruth_list))
        return category_dict


class ImageNetTools(object):
    """
    Define the ImageNet tools, synsets reading and loading.
    """
    def __init__(self, fpath):
        meta_clsloc_file = fpath
        self.synsets = loadmat(meta_clsloc_file)["synsets"][0]
        self.synsets_imagenet_sorted = sorted(
            [(int(s[0]), str(s[1][0])) for s in self.synsets[:1000]], key=lambda v: v[1])
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
    f.create_dataset('category_dict', data=imageNetLoader.classes_indices,
                     compression='gzip', compression_opts=9)
    f.create_dataset('valid_groundtruth_dict', data=imageNetLoader.valid_groundtruth_dict,
                     compression='gzip', compression_opts=9)


if __name__ == '__main__':
    # save_list_to_h5df()
    data = ImageNetData('/home/kyu/.keras/datasets/ILSVRC2015')
    TARGET_SIZE = (224, 224)
    RESCALE_SMALL = (256, 512)
    gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                     horizontal_flip=True,
                                     )
    valid_gen = ImageDataGeneratorAdvanced(TARGET_SIZE,
                                           rescaleshortedgeto=256,
                                           random_crop=False,
                                           horizontal_flip=True)

    def label_wrapper(label):
        imagenettool = data.imagenettool
        return imagenettool.id_to_words(imagenettool.synset_to_id(label))

    valid = data.get_test(image_data_generator=valid_gen, save_to_dir='/home/kyu/plots',
                          save_prefix='imagenet_valid', save_format='JPEG', label_wrapper=label_wrapper,
                          shuffle=False)

    train = data.get_train(image_data_generator=gen, save_to_dir='/home/kyu/plots',
                           save_prefix='imagenet', save_format='JPEG', label_wrapper=label_wrapper,
                           shuffle=False)

    # a, b = train.next()
    a, b = valid.next()
    # a, b = train.next()
