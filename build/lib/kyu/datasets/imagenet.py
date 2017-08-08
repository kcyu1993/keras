# -*- coding: utf-8 -*-

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from random import shuffle

import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array, array_to_img

import numpy as np
import glob
import logging

from scipy.io import loadmat
from scipy.misc import imresize

from kyu.utils.image import crop_img, random_crop_img, ImageDataGeneratorAdvanced
## Imagenet

IMAGENET_VALID_GROUNDTRUTH_FILE = 'ILSVRC2014_clsloc_validation_ground_truth.txt'
IMAGENET_VALID_BLACKLIST_FILE = 'ILSVRC2014_clsloc_validation_blacklist.txt'


def preprocess_image_for_imagenet(img):
    """
     preprocessing_function: function that will be implied on each input.

            The function will run before any other modification on it.
            The function should take one argument: one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.

    Parameters
    ----------
    img : ndarray with rank 3

    Returns
    -------
    img : ndarray with same shape
    """

    dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    x = img
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x


# class ImageDataGeneratorAdvanced(ImageDataGenerator):
#     """
#     Advanced operation:
#         Support ImageNet training data.
#
#     """
#     def __init__(self,
#                  rescaleshortedgeto=None,
#                  random_crop=False,
#                  target_size=None,
#                  **kwargs):
#         self.random_crop = random_crop
#         self.rescaleshortedgeto = rescaleshortedgeto
#         self.target_size = target_size
#         super(ImageDataGeneratorAdvanced, self).__init__(**kwargs)
#
#     def advancedoperation(self, x):
#         if self.rescaleshortedgeto:
#             pass
#
#         if self.random_crop and self.target_size is not None:
#             # Implement random crop
#             pass
#
#     # def random_crop(self, x, target_size, dim_ordering='default'):
#     #     if dim_ordering == 'default':
#     #         dim_ordering = K.image_dim_ordering()
#     #     if dim_ordering == 'tf':
#     #         if len(x.shape) == 2:
#     #             widthchannel = 0
#     #             heightchannel = 1
#     #         elif len(x.shape) == 3:
#     #             widthchannel = 0
#     #             heightchannel = 1
#     #         # elif len(x.shape) == 4:
#     #         #     widthchannel = 1
#     #         #     heightchannel = 2
#     #         else:
#     #             raise ValueError
#     #     else:
#     #         if len(x.shape) == 2:
#     #             widthchannel = 0
#     #             heightchannel = 1
#     #         elif len(x.shape) == 3:
#     #             widthchannel = 1
#     #             heightchannel = 2
#     #         # elif len(x.shape) == 4:
#     #         #     widthchannel = 2
#     #         #     heightchannel = 3
#     #         else:
#     #             raise ValueError
#     #
#     #     # Target size
#     #     width = x.shape[widthchannel]
#     #     height = x.shape[heightchannel]
#     #     w_limit = width - target_size[0]
#     #     h_limit = height - target_size[1]
#     #     rand_w = np.random.randint(0, w_limit)
#     #     rand_h = np.random.randint(0, h_limit)
#     #     crop
#     #     return


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


class ImageIterator(Iterator):
    """
    Define a general iterator.
    Given a image files, iterate through it.
    """
    def __init__(self, imgloc_list, cate_list, nb_class,
                 image_data_generator=None,
                 dir_prefix='',
                 target_size=(224,224), color_mode='rgb',
                 dim_ordering='default',
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 category_dict=None,
                 save_to_dir=None, save_prefix='', save_format='JPEG'):
        """
        Create a ImageIterator based on image locations and corresponding categories.
        One should obtain all location regarding to the images before use the iterator.
        Use next() to generate a batch.

        Parameters
        ----------
        imgloc_list : list      contains image paths list
        cate_list : list        contains the corresponding categories regarding to the
        image_data_generator    Generator of keras
        dir_prefix : str        Prefix of path to images
        target_size : (int,int)
        color_mode : str        rgb, grayscale
        dim_ordering : str      th, tf
        class_mode : str        category, sparse, binary
        batch_size : int
        shuffle : bool
        seed : Not used here
        save_to_dir : str       For test purpose.
        save_prefix
        save_format
        """
        assert len(imgloc_list) == len(cate_list)

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering

        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.nb_class = int(nb_class)
        self.dir_prefix = dir_prefix

        # Generator settings
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Image settings
        self.target_size = target_size
        self.color_mode = color_mode
        self.imageOpAdv = False
        if image_data_generator is None:
            image_data_generator = ImageDataGenerator(horizontal_flip=True)
        elif isinstance(image_data_generator, ImageDataGeneratorAdvanced):
            self.imageOpAdv = True
            image_data_generator.target_size = self.target_size
        self.image_data_generator = image_data_generator

        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        # Test purpose
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        # Generate the image list and corresponding category
        self.img_files_list = [self.get_image_path(i) for i in imgloc_list]
        self.img_cate_list = cate_list.astype(np.uint32)
        self.category_dict = category_dict

        self.nb_sample = len(self.img_files_list)
        super(ImageIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
        # self.index_generator = self._flow_index(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'

        # Build image data
        for i, j in enumerate(index_array):
            fname = self.img_files_list[j]
            img = load_img(fname, grayscale=grayscale)
            # Random crop
            # img = random_crop_img(img, target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)

            # x = preprocess_input(x, dim_ordering=self.dim_ordering)
            if self.imageOpAdv:
                x = self.image_data_generator.advancedoperation(x)
            else:
                x = imresize(x, self.target_size)
                if self.dim_ordering == 'th':
                    x = x.transpose(2,0,1)
            x = x.astype(K.floatx())
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            batch_x[i] = x


        # build batch of labels
        if self.class_mode == 'sparse':
            raise NotImplementedError
            # batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            raise NotImplementedError
            # batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, j in enumerate(index_array):
                label = self.img_cate_list[j]
                # label = self.img_cate_list[j] - 1
                batch_y[i, label] = 1.
        else:
            return batch_x

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                if self.category_dict is not None:
                    label = 'None'
                    for (cate_name, ind) in self.category_dict.items():
                        if ind == self.img_cate_list[index_array[i]]:
                            label = cate_name
                    fname = '{prefix}_{index}_{hash}-{label}.{format}'.format(
                        prefix=self.save_prefix,
                        index=current_index + i,
                        hash=np.random.randint(1e4),
                        format=self.save_format,
                        label=label + str(self.img_cate_list[index_array[i]])
                    )
                else:
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=current_index + i,
                        hash=np.random.randint(1e4),
                        format=self.save_format,
                    )
                img.save(os.path.join(self.save_to_dir, fname))

        batch_x = preprocess_input(batch_x, dim_ordering=self.dim_ordering)

        return batch_x, batch_y

    def get_image_path(self, img_file):
        """ Get Image img_file """
        return os.path.join(self.dir_prefix, img_file)


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
                print (str(e))
                load_train = True
            try:
                self.valid_list = fdata['valid_list']
            except KeyError as e:
                print (str(e))
                load_valid = True
            try:
                self.test_list = fdata['test_list']
            except KeyError as e:
                print (str(e))
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
    # IMAGENET_PATH = '/home/kyu/.keras/datasets/ILSVRC2015'
    # TARGET_SIZE = (224, 224)
    # RESCALE_SMALL = 256
    #
    # BATCH_SIZE = 16
    # NB_EPOCH = 70
    # VERBOSE = 1
    # SAVE_LOG = False
    #
    # # ImageNet generator
    # imageNetLoader = ImageNetLoader(IMAGENET_PATH)
    # gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
    #                                  horizontal_flip=True,
    #                                  channelwise_std_normalization=True)
    #
    # train = imageNetLoader.generator('train', image_data_generator=gen)
    # valid = imageNetLoader.generator('valid', image_data_generator=gen)
    # # test = imageNetLoader.generator('valid', image_data_generator=gen)
    # HHA
    save_list_to_h5df()
