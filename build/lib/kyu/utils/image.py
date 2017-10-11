import os

import numpy as np
from PIL import Image
from scipy.misc import imresize

import keras.backend as K
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator, Iterator, load_img, img_to_array, \
    array_to_img
from kyu.utils.dict_utils import create_dict_by_given_kwargs
from kyu.utils.imagenet_utils import preprocess_image_for_imagenet_without_channel_reverse, \
    preprocess_image_for_imagenet_of_densenet, preprocess_image_for_imagenet


def crop(x, center_x, center_y, ratio=.23, channel_index=0):
    """

    :param x: as nparray (3,x,x)
    :param center_x: float  [0,1]
    :param center_y: float  [0,1]
    :param ratio: float     [0,1]
    :return:
    """
    ratio = max(0,ratio)
    ratio = min(ratio,1)
    assert len(x.shape) == 3 and x.shape[channel_index] in {1, 3}
    r_x = [max(0, center_x - ratio/2), min(1, center_x + ratio/2)]
    r_y = [max(0, center_y - ratio/2), min(1, center_y + ratio/2)]
    if channel_index == 0:
        w = x.shape[1]
        h = x.shape[2]
        return x[:,int(r_x[0]*w):int(r_x[1]*w),int(r_y[0]*h):int(r_y[1]*h)]
    elif channel_index == 2:
        w = x.shape[0]
        h = x.shape[1]
        return x[int(r_x[0] * w):int(r_x[1] * w), int(r_y[0] * h):int(r_y[1] * h), :]
    else:
        raise ValueError("Only support channel as 0 or 2")


def crop_img(img, x, y, ratio=.23, target_size=None):
    ratio = max(0, ratio)
    ratio = min(ratio, 1)
    r_x = [max(0, x - ratio / 2), min(1, x + ratio / 2)]
    r_y = [max(0, y - ratio / 2), min(1, y + ratio / 2)]
    w, h = img.size
    img = img.crop((int(r_x[0] * w), int(r_y[0] * h), int(r_x[1] * w), int(r_y[1] * h)))

    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def random_crop_img(img, target_size):
    """
    Random crop the image to target-size.

    Parameters
    ----------
    img
    target_size

    Returns
    -------

    """
    w,h = img.size
    if not(w >= target_size[0] and h >= target_size[1]):
        return img
    cor_x = np.random.randint(0, w - target_size[0])
    cor_y = np.random.randint(0, h - target_size[1])
    img = img.crop((cor_x, cor_y, cor_x + target_size[0], cor_y + target_size[1]))
    return img


class DirectoryIteratorWithFile(DirectoryIterator):
    """
    Directory with the text file

    """
    def __init__(self, directory, container_file,
                 image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        """

        :param directory:               root directory
        :param container_file:          absolute path, contains " path/under/root/to/image.type /n ... "
        :param image_data_generator:
        :param target_size:
        :param color_mode:
        :param dim_ordering:
        :param classes:
        :param class_mode:
        :param batch_size:
        :param shuffle:
        :param seed:
        :param save_to_dir:
        :param save_prefix:
        :param save_format:
        """
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.container_file = container_file
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color cov_mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
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
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            raise NotImplementedError

        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        container_path = container_file
        with(open(container_path, 'r')) as f:
            file_list = [line.rstrip() for line in f]
            f.close()
        # for fname in sorted(file_list):
        #     is_valid = False
        #     for extension in white_list_formats:
        #         if fname.lower().endswith('.' + extension) and os.path.exists(os.path.join(directory, fname)):
        #             is_valid = True
        #             break
        #     if is_valid:
        #         self.nb_sample += 1
        self.nb_sample = len(file_list)
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')

        i = 0
        for fname in sorted(file_list):
            is_valid = False
            _, file_name = os.path.split(fname)
            class_name, _ = file_name.split("_")
            # for extension in white_list_formats:
            #     if fname.lower().endswith('.' + extension) and os.path.exists(os.path.join(directory, fname)):
            #         is_valid = True
            #         break
            # if is_valid:
            self.classes[i] = self.class_indices[class_name]
            self.filenames.append(os.path.join(directory, fname))
            i += 1

        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)


class ImageDataGeneratorAdvanced(ImageDataGenerator):
    """
    Advanced operation:
        Support ImageNet training data.

    """

    @classmethod
    def get_default_train_config(cls):
        return create_dict_by_given_kwargs(
            rescaleshortedgeto=(256, 296), random_crop=True, horizontal_flip=True)

    @classmethod
    def get_default_valid_config(cls):
        return create_dict_by_given_kwargs(
            rescaleshortedgeto=256, random_crop=False, horizontal_flip=True)

    def __init__(self,
                 target_size=None,
                 rescaleshortedgeto=None,
                 random_crop=False,
                 **kwargs):
        """
        Target size is allowed to be altered during training process.

        Parameters
        ----------
        target_size
        rescaleshortedgeto
        random_crop
        kwargs
        """
        self.random_crop = random_crop
        self.rescaleshortedgeto = rescaleshortedgeto
        self.target_size = target_size
        super(ImageDataGeneratorAdvanced, self).__init__(**kwargs)

    def advancedoperation(self, x, data_format='default'):
        """
        Make the image to target size based on operations.

        Parameters
        ----------
        x : ndarray (3, w, h)

        Returns
        -------
        ndarray (3, target_size) or (target_size, 3)
        """
        if data_format == 'default':
            data_format = self.data_format
        # Change back to channels last.
        if data_format == 'channels_last':
            x = x.transpose(2,0,1)

        width = x.shape[1]
        height = x.shape[2]
        aspect_ratio = float(width) / float(height)
        cornor = [0.0, 0.0]

        # Rescale the image to target-size
        new_width = width if width > self.target_size[0] else self.target_size[0]
        new_height = width / aspect_ratio
        if new_height < self.target_size[1]:
            new_height = self.target_size[1]
            new_width = new_height * aspect_ratio

        if isinstance(self.rescaleshortedgeto, (list, tuple)):
            assert len(self.rescaleshortedgeto) == 2
            rescaleshortedgeto = np.random.randint(self.rescaleshortedgeto[0], self.rescaleshortedgeto[1])
        elif isinstance(self.rescaleshortedgeto, int):
            rescaleshortedgeto = self.rescaleshortedgeto
        else:
            rescaleshortedgeto = None

        if self.rescaleshortedgeto:

            short_edge = min(new_height, new_width)
            short_aspect_ratio = float(max(new_height, new_width))/\
                                 float(min(new_height, new_width))
            if new_width > new_height:
                new_height = rescaleshortedgeto
                new_width = new_height * aspect_ratio
            else:
                new_width = rescaleshortedgeto
                new_height = new_width / aspect_ratio
        # Note the image resize only supports tf like image shape
        # x = x.transpose((1,2,0))
        tx = imresize(x, [int(new_width), int(new_height)]).transpose(2,0,1)
        if self.random_crop:
            # Cornor change
            cornor[0] = np.random.randint(0, int(new_width - self.target_size[0]))
            cornor[1] = np.random.randint(0, int(new_height - self.target_size[1]))
        else:
            # Center the crop:
            cornor[0] = int((new_width - self.target_size[0]) / 2)
            cornor[1] = int((new_height - self.target_size[1]) / 2)
        tx = tx[
               :,
               cornor[0]: cornor[0] + self.target_size[0],
               cornor[1]: cornor[1] + self.target_size[1]
               ]
        if data_format == 'channels_last':
            tx = tx.transpose(1,2,0)
        return tx


class ClassificationIterator(Iterator):
    """
    Give a list of image, and its corresponding classes, provide generator.
    """

    def __init__(self, img_list, class_list,
                 image_data_generator,
                 load_path_prefix='',
                 target_size=(224,224), color_mode='rgb',
                 data_format='default',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 class_mode='categorical',
                 follow_links=False
                 ):
        """

        Parameters
        ----------
        img_list : list[str]        list of paths to images
        class_list  : list[int]     list of int, starting from 1
        image_data_generator :      ImageDataGenerator
        load_path_prefix :          prefix of loading paths
        target_size
        color_mode
        dim_ordering
        batch_size
        shuffle
        seed
        save_to_dir
        save_prefix
        save_format
        follow_links
        """
        if data_format == 'default':
            data_format = K.image_data_format()
        if image_data_generator is None:
            image_data_generator = ImageDataGenerator()
        self.load_path_prefix = load_path_prefix

        self.image_data_generator = image_data_generator

        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode

        # Update Image data informations
        self.img_list = np.asanyarray(img_list)
        self.classes = np.asanyarray(class_list)

        self.nb_sample = np.max(self.img_list.shape)
        self.nb_class = np.max(np.unique(self.classes).shape)
        super(ClassificationIterator, self).__init__(self.nb_sample, batch_size, shuffle=shuffle, seed=seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.img_list[j]
            img = load_img(os.path.join(self.load_path_prefix, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class ImageLoader(object):

    def __init__(self, dirpath):
        """Create General ImageLoader related operations """
        self.r_dirpath = dirpath

        self.categories = None
        self.label = []
        self.image = []

        self.image_dir = None
        self.label_dir = None


    def _load(self, fpath):
        """
                Load single image from given path
                :param fpath: full path.
                :return:      class label, image 3D array
                """
        dir, filename = os.path.split(fpath)
        index, _ = self.decode(filename)
        fpath = self.getimagepath(fpath)
        img = ImageLoader.load_image(fpath)
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
            data = data.transpose([2, 0, 1])
        elif len(data.shape) == 2:
            data = np.ones((3, data.shape[0], data.shape[1])) * data
        return data

    """ Abstract methods """
    def getimagepath(self, fpath):
        pass

    def decode(self, filename):
        pass


def get_densenet_image_gen(target_size, **kwargs):
    return ImageDataGeneratorAdvanced(
        target_size, preprocessing_function=preprocess_image_for_imagenet_of_densenet, **kwargs
    )


def get_vgg_image_gen(target_size, **kwargs):

    return ImageDataGeneratorAdvanced(
        target_size, preprocessing_function=preprocess_image_for_imagenet_without_channel_reverse, **kwargs
        # preprocessing_function=preprocess_image_for_imagenet
    )


def get_resnet_image_gen(target_size, **kwargs):
    return ImageDataGeneratorAdvanced(
        target_size, preprocessing_function=preprocess_image_for_imagenet_without_channel_reverse, **kwargs
    )


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
                 data_format='default',
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 category_dict=None,
                 save_to_dir=None, save_prefix='', save_format='JPEG',
                 label_wrapper=None):
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

        # Legacy support
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.dim_ordering = dim_ordering

        self.data_format = K.image_data_format() if data_format == 'default' else data_format

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
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        # Test purpose
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.label_wrapper = label_wrapper

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
            x = img_to_array(img, data_format=self.data_format)

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
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                if self.category_dict is not None:
                    label = 'None'
                    for (cate_name, ind) in self.category_dict.items():
                        if ind == self.img_cate_list[index_array[i]]:
                            label = cate_name
                    # Use label wrapper
                    if self.label_wrapper is not None:
                        label = self.label_wrapper(label)
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

        # batch_x = preprocess_input(batch_x, data_format=self.data_format)

        return batch_x, batch_y

    def get_image_path(self, img_file):
        """ Get Image img_file """
        return os.path.join(self.dir_prefix, img_file)