# -*- coding: utf-8 -*-
import gzip
from ..utils.data_utils import get_file, get_file_from_dir, get_dataset_dir
from ..utils.io_utils import HDF5Matrix
import numpy as np
import cPickle
import sys
from PIL import Image
import os.path
from os import listdir
import h5py
from keras.preprocessing.image import DirectoryIteratorWithFile, DirectoryIterator, ImageDataGenerator


class MincLoader(object):
    """
    MINC dataset loader
    Created for both loading minc2500 and minc-original
    """
    def __init__(self, dirpath='minc-2500', category='categories.txt',
                 label_dir='labels', image_dir='images'):
        # self.__name__ = 'MINCloader :'
        self.r_dirpath = dirpath
        # print(self.__name__ + dirpath)
        self.abs_dirpath = os.path.join(get_dataset_dir(), dirpath)
        self.image_dir = os.path.join(self.abs_dirpath, image_dir)
        self.labels_dir = os.path.join(self.abs_dirpath, label_dir)


        # Create and store the categories.
        self.categories = self.loadcategory(category)

        # create the holders
        self.label = []
        self.image = []

    def _load(self, fpath):
        """
        Load single image from given path
        :param fpath: full path.
        :return:      class label, image 3D array
        """
        dir, filename = os.path.split(fpath)
        index,_ = self.decode(filename)
        fpath = self.getimagepath(fpath)
        img = MincLoader.load_image(fpath)
        return index, img

    @staticmethod
    def load_image(fpath):
        """
        Load the jpg image as np.ndarray(3, row, col)
        Following the theano image ordering.

        :param fpath:
        :return:
        """
        img = Image.open(fpath)
        img.load()
        data = np.asarray(img, dtype="byte")
        if len(data.shape) == 3:
            data = data.transpose([2,0,1])
        elif len(data.shape) == 2:
            data = np.ones((3, data.shape[0], data.shape[1])) * data

        return data

    def loadcategory(self, file=None):
        """
        Load category list and generate the
        :param file:
        :return: list.
        """
        # print(self.abs_dirpath + file)
        fpath = get_file(os.path.join(self.abs_dirpath, file), origin=None)
        with open(fpath, 'r') as f:
            return dict((name, ind) for ind, name in enumerate([line.rstrip() for line in f]))

    def decode(self, filename):
        """
        Decode the file name in labels file.
        :param filename: example name "brick_001578.jpg"
        :return class_index, img_index
        """
        key, ind = filename.split('_')
        ind, _ = ind.split('.')
        return self.categories.get(key), ind

    def getimagepath(self, filepath, **kwargs):
        raise NotImplementedError

    def loadfromfile(self, filename='minc-2500.plk.gz'):
        path = os.path.join(self.abs_dirpath, filename)
        if os.path.exists(path):
            if path.endswith(".gz"):
                f = gzip.open(path, 'rb')
            elif path.endswith("hdf5"):
                # Old loading into nparray
                # f = h5py.File(os.path.join(self.abs_dirpath, filename), "r")
                # tdata = []
                # # group = f.get('data')
                # for ind, name in enumerate(self.hd5f_label):
                #     # tdata[ind] = group[name[ind]]
                #     tdata.append(f.get(name).value)
                # data = (tdata[0], tdata[1]), (tdata[2], tdata[3]), (tdata[4], tdata[5])
                # return data
                #####

                # load into HDF5Matrix
                tdata = []
                for ind, name in enumerate(self.hd5f_label):
                    print("normalized version")
                    data = HDF5Matrix(path, name, normalizer=lambda x: x*1.0/255)
                    tdata.append(data)
                data = (tdata[0], tdata[1]), (tdata[2], tdata[3]), (tdata[4], tdata[5])
                return data
            else:
                f = open(path, 'rb')
            # f = open(path, 'rb')
            if sys.version_info < (3,):
                data = cPickle.load(f)
            else:
                raise NotImplementedError
                # data = cPickle.load(f, encoding="bytes")

            f.close()
            return data # (data
        else:
            print("File not found under location {}".format(path))
            return None

    def save(self, data=None, output_file='minc-2500.plk', ftype='gz'):
        """
        Save the self.label, self.image before normalization, to Pickle file
        To the specific output directory.
        :param data: 3*2 tuple: (tr_img, tr_lab) .. for valid, test.
        :param output_file:
        :param ftype
        :return:
        """
        if data is None:
            data = (self.image, self.label)
            output_file = 'minc-2500-whole.plk'

        if ftype is 'gz' or ftype is 'gzip':
            print("compress with gz")
            f = gzip.open(os.path.join(self.abs_dirpath, output_file + "." + ftype), 'wb')
        elif ftype is '':
            f = open(os.path.join(self.abs_dirpath, output_file), 'wb')
        elif ftype is 'hdf5':
            f = h5py.File(os.path.join(self.abs_dirpath, output_file + "." + ftype), "w")
            # arr = f.create_group('minc', data, chunks=True)
            # group = f.create_group('data')
            for ind, name in enumerate(self.hd5f_label):
                f.create_dataset(name, data=data[ind/2][ind%2], compression="gzip", compression_opts=9)
            return
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


class Minc2500(MincLoader):
    """
    Loader for MINC image dataset

    Part 1
    Logic:
        give dir-path
        load the categorty.txt first, translated as LookUpTable (index them)


    MINC patch information: http://opensurfaces.cs.cornell.edu/static/minc/minc.tar.gz
    MINC-2500: http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz

    MINC-2500 structure:

    path/to/minc-2500
        categories.txt
        images/
            cate_name/
                cate_name_index.jpg
                ...
        labels/
            trainN.txt
            validN.txt
            testN.txt
        minc-2500.plk.gz
            cPickle version.


    """
    def __init__(self, dataset=1, dirpath='minc-2500', category='categories.txt',
                 label_dir='labels', image_dir='images'):
        self.data_index = dataset
        self.hd5f_label = ['tr_img', 'tr_lab', 'va_img', 'va_lab', 'te_img', 'te_lab']
        super(Minc2500, self).__init__(dirpath='minc-2500', category='categories.txt',
                                       label_dir='labels', image_dir='images')

    def load(self, names=None, overwrite=True, split=True, index=1):
        if names is None:
            names = self.categories.keys()

        if split:
            return self.loadwithsplit(names, index)

        if overwrite:
            self.label = []
            self.image = []
        for name in names:
            # get all list from the folder
            name_dir = os.path.join(self.image_dir, name)
            list = os.listdir(name_dir)
            for file in list:
                index, img = self._load(file)
                self.label.append(index)
                self.image.append(img)

        # load the category and generate the look up table
        return np.array(self.label), np.array(self.image)

    def loadwithsplit(self, names=None, index=1):
        """
        Load with split of train, validate and test datasets
        :param names:
        :param index:
        :return:
        """
        if index < 1 or index > 5:
            print("Index out of range")
            return None

        if names is None:
            names = self.categories.keys()
        print("Loading with split ")
        tr_path = os.path.join(self.labels_dir, "train{}.txt".format(index))
        va_path = os.path.join(self.labels_dir, "validate{}.txt".format(index))
        te_path = os.path.join(self.labels_dir, "test{}.txt".format(index))
        with(open(tr_path, 'r')) as f:
            tr_list = [line.rstrip().split('/')[2] for line in f]
            f.close()
        with(open(va_path, 'r')) as f:
            va_list = [line.rstrip().split('/')[2] for line in f]
            f.close()
        with(open(te_path, 'r')) as f:
            te_list = [line.rstrip().split('/')[2] for line in f]
            f.close()

        tr_label = []
        te_label = []
        va_label = []
        tr_img = []
        te_img = []
        va_img = []
        # tr_label = np.array()
        # te_label = np.array()
        # va_label = np.array()
        # tr_img = np.array()
        # te_img = np.array()
        # va_img = np.array()
        print("Load training {}".format(len(tr_list)))
        # for file in tr_list:
        for file in tr_list[:100]:
            index, img = self._load(file)
            tr_label.append(index)
            tr_img.append(img)
        print("Load validate {}".format(len(va_list)))
        # for file in va_list:
        for file in va_list[:100]:
            index, img = self._load(file)
            va_label.append(index)
            va_img.append(img)
        print("Load test {}".format(len(te_list)))
        # for file in te_list:
        for file in te_list[:100]:
            index, img = self._load(file)
            te_label.append(index)
            te_img.append(img)
        print("load finish with {} train, {} valid and {} test".format(len(tr_label),
                                                                       len(va_label),
                                                                       len(te_label)))

        # return the np array as required.
        return (np.array(tr_img), np.array(tr_label)), \
               (np.array(va_img), np.array(va_label)), \
               (np.array(te_img), np.array(te_label))

    def getimagepath(self, fpath, index=None):
        """
        Get image absolute path.
        :param fpath:   cate_index.*
        :param index:
        :return:    absolute path to image
        """
        cate_name,_ = fpath.split('_')
        dir = os.path.join(self.image_dir, cate_name)
        return os.path.join(dir, fpath)

    def generator(self, file='train1.txt', shuffle=True, batch_size=32):
        gen = ImageDataGenerator(rescale=1./255)
        print("generate the image with batch size {} shuffle {}".format(batch_size, shuffle))
        if file is None:
            iterator = DirectoryIterator(self.image_dir, gen, shuffle=shuffle, target_size=(362,362),
                                         batch_size=batch_size, classes=self.categories)
        else:
            fpath = os.path.join(self.labels_dir, file)
            iterator = DirectoryIteratorWithFile(self.abs_dirpath, fpath, gen, shuffle=shuffle, target_size=(362,362),
                                                 batch_size=batch_size, classes=self.categories)
        # batch_x, batch_y = iterator.next()
        return iterator

    def test_hd5f(self):
        tr = [np.random.rand(1000,3,362,362), np.array(range(10), dtype='byte')]
        va = np.array(range(11, 15), dtype='byte')
        te = np.array(range(16, 20), dtype='byte')
        f = h5py.File(os.path.join(self.abs_dirpath, 'test.hdf5'), "w")
        # group = f.create_group("minc2500")
        # train = group.create_group('train')
        # test = group.create_group('test')
        # valid = group.create_group('valid')
        # f.create_dataset('minc', data=np.array((tr, va, te)))
        # group['train'] = tr
        # group['test'] = te
        # group['valid'] = va

    def test_hd5f_load(self):
        f = h5py.File(os.path.join(self.abs_dirpath, 'test.hdf5'), "r")
        print(f.name)
        # group = f.get('minc2500')
        # tr = group['train']
        # va = group['valid']
        # te = group['test']
        data = f.get('minc').value
        return data


class MincOriginal(MincLoader):
    """
    MincOriginal
        Author: Kaicheng Yu     2016-11-7
    Function:
        Load the raw image from minc dataset
        Create the corresponding PATCH for each image, with fix size (360, 360)
            By bounding box centered at click, size is 23% lower dimension of the image
            Upscaling / Downscaling to 360,360
        Once determined, further save as pickle file

    Data structure as output:
        data = (nbsample, 3, 360, 360)
        label = (nbsample, )
        (train_data, train_label) , ... for valid, test.

    Structure:
    */minc/
        photo_orig/
            0/      __id__0.jpg
            1/      __id__1.jpg
            ...
            9/      __id__9.jpg
            (each of folders contains the photo whose id ends with folder number)
        minc-s/ # NOT FOR THIS MOMENT
            segments/ (mask, 1 (white) as true, 0 black as false)
                PHOTOID_PATCHID.png
            categories.txt
            README.txt
            test-clicks.txt
                22,000034234,0.0859951062148877,0.297752808988764
            test-segments.txt

        categories.txt
        README.txt
        train.txt           A 4-tuple list of (label, photo id, x, y). Point locations are normalized to be in the range [0, 1].
        validate.txt        same
        test.txt            same


    """

    def __init__(self):
        raise NotImplementedError

    def _load(self, fpath, x, y):
        """
        Load single image from given path.
        :param fpath: full path.
        :return:      class label, image 3D array
        """
        dir, filename = os.path.split(fpath)
        index,_ = self.decode(filename)
        fpath = self.getimagepath(fpath)
        img = MincLoader.load_image(fpath)

        return index, img
