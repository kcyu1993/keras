import numpy as np
from scipy.misc import imresize

from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator, Iterator, load_img, img_to_array, \
    array_to_img
import keras.backend as K
import os


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


class MincOriginalIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 classes, txtfile='train.txt', ratio=.23,
                 nb_per_class=10000,
                 img_folder='photo_orig',
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 class_mode='categorical',
                 batch_size=32, shuffle=True, seed=222,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
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

        self.img_folder = img_folder
        self.txtfile = txtfile
        self.ratio = ratio
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        self.batch_array = []

        self.nb_class = len(classes)
        self.class_indices = {v: k for k, v in classes.iteritems()}

        # Specific for Minc Original Structure
        # dirs = range(10)
        # for subdir in dirs:
        #     subpath = os.path.join(self.img_folder, str(subdir))
        #     for fname in sorted(os.listdir(subpath)):
        #         is_valid = False
        #         for extension in white_list_formats:
        #             if fname.lower().endswith('.' + extension):
        #                 is_valid = True
        #                 break
        #         if is_valid:
        #             self.nb_sample += 1

        # second, build an index of the images in the different class subfolders
        self.classeslist = self.generatelistfromtxt(txtfile)
        self.nb_sample = 0
        for l in self.classeslist:
            self.nb_sample += len(l)
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # Specify nb_per_class
        self.nb_per_class = nb_per_class \
            if nb_per_class < min(map(len, self.classeslist)) \
            else min(map(len, self.classeslist))
        self.nb_sample = self.nb_per_class * self.nb_class
        super(MincOriginalIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
        self.index_generator = self._flow_index(self.nb_per_class * self.nb_class, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            # get the index
            # print(self.batch_array[j])
            fname = self.batch_array[j][0]
            center_x, center_y = self.batch_array[j][1:]
            img = load_img(os.path.join(self.img_folder, fname), grayscale=grayscale)
            img = crop_img(img, center_x, center_y, ratio=self.ratio, target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        batch_label = []

        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.batch_classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.batch_classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate([self.batch_classes[i] for i in index_array]):
                batch_y[i, label] = 1.
                batch_label.append(self.class_indices[label])
        else:
            return batch_x
        # optionally save augmented images to disk for debugging purposes

        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix + \
                                                                  batch_label[i],
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)

                img.save(os.path.join(self.save_to_dir, fname))
        return batch_x, batch_y

    def generatelistfromtxt(self, fname):
        if self.classes is None:
            raise ValueError("classes should be initialized before calling")

        path = fname
        # label, photo_id, x, y
        with(open(path, 'r')) as f:
            file_list = [line.rstrip().split(',') for line in f]
            f.close()
        # load the category and generate the look up table
        classlist = [[] for i in range(self.nb_class)]
        for i, fname, x, y in file_list:
            fn = os.path.join(fname.split('.')[0][-1], fname + ".jpg")
            classlist[int(i)].append([fn, float(x), float(y)])
        return classlist

    def reset(self):
        """
        reset the batch_array and index
        :return:
        """
        self.batch_index = 0
        self.batch_array = []
        self.batch_classes = []
        # Generate balanced sample for each epoch
        for i in range(self.nb_class):
            index_array = np.arange(len(self.classeslist[i]))
            # if self.shuffle:
            #     index_array = np.random.permutation(len(self.classeslist[i]))

            array = index_array[:self.nb_per_class]
            self.batch_array += ([self.classeslist[i][a] for a in array])
            self.batch_classes += ([i for j in range(self.nb_per_class)])

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        """
        flow for the Minc Original dataset
        Create a random 10,000 per class, total 230,000 per iteration

        :param N:
        :param batch_size:
        :param shuffle:
        :param seed:
        :return:
        """
        # Special flow_index for the balanced training sample generation
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)



class ImageDataGeneratorAdvanced(ImageDataGenerator):
    """
    Advanced operation:
        Support ImageNet training data.

    """
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

    def advancedoperation(self, x, dim_ordering='default'):
        """
        Make the image to target size based on operations.

        Parameters
        ----------
        x : ndarray (3, w, h)

        Returns
        -------
        ndarray (3, target_size) or (target_size, 3)
        """
        if dim_ordering == 'default':
            dim_ordering = self.dim_ordering
        # Change back to th.
        if dim_ordering == 'tf':
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

        if self.rescaleshortedgeto:

            short_edge = min(new_height, new_width)
            short_aspect_ratio = float(max(new_height, new_width))/\
                                 float(min(new_height, new_width))
            if new_width > new_height:
                new_height = self.rescaleshortedgeto
                new_width = new_height * aspect_ratio
            else:
                new_width = self.rescaleshortedgeto
                new_height = new_width / aspect_ratio
        # Note the image resize only supports tf like image shape
        # x = x.transpose((1,2,0))
        tx = imresize(x, [int(new_width), int(new_height)]).transpose(2,0,1)
        if self.random_crop:
            # Cornor change
            cornor[0] = np.random.randint(0, new_width - self.target_size[0])
            cornor[1] = np.random.randint(0, new_height - self.target_size[1])

        tx =  tx[
               :,
               cornor[0]: cornor[0] + self.target_size[0],
               cornor[1]: cornor[1] + self.target_size[1]
               ]
        if dim_ordering == 'tf':
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
                 dim_ordering='default',
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
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if image_data_generator is None:
            image_data_generator = ImageDataGenerator()
        self.load_path_prefix = load_path_prefix

        self.image_data_generator = image_data_generator

        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
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
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
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