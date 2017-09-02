import os

import numpy as np

from keras import backend as K
from keras.preprocessing.image import Iterator, load_img, img_to_array, array_to_img
from kyu.utils.image import crop_img


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