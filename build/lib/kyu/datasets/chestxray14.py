"""
Implement the loading for Chest Xray dataset.
Currently for classification tasks

"""
import csv
import os
import numpy as np
import h5py
import sys

from keras.preprocessing.image import Iterator, load_img, img_to_array
from kyu.engine.utils.data_utils import ClassificationImageData, BoundingBox
from kyu.utils.dict_utils import create_dict_by_given_kwargs, load_dict
import json
import keras.backend as K
from kyu.utils.image import ImageDataGeneratorAdvanced


def chestxray_labels():
    return [u'Atelectasis', u'Cardiomegaly', u'Consolidation', u'Edema', u'Effusion', u'Emphysema', u'Fibrosis', u'Hernia', u'Infiltration', u'Mass', u'Nodule', u'Pleural_Thickening', u'Pneumonia', u'Pneumothorax']


def preprocess_image_for_chestxray(img):
    """
        Preprocess Image for CUB dataset

        ,
        ,
        108.68376923,
        :param img: ndarray with rank 3
        :return: img: ndarray with same shape

        """
    mean = 108
    data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}
    x = img
    if data_format == 'channels_first':
        # Zero-center by mean pixel
        x[0, :, :] -= mean
        x[1, :, :] -= mean
        x[2, :, :] -= mean
    else:
        # Zero-center by mean pixel
        x[:, :, 0] -= mean
        x[:, :, 1] -= mean
        x[:, :, 2] -= mean
    return x


def make_inference_dataset():
    path = '/cvlabdata1/cvlab/datasets_kyu/chest-xray14'
    # Open the h5file
    os.chdir('/cvlabdata1/cvlab/datasets_kyu/chest-xray14')
    filename = 'BBox_List_2017.csv'

    with open(os.path.join(path, filename), 'rb') as f:
        reader = csv.reader(f)
        title = reader.next()
        entry = []
        for row in reader:
            entry.append(row)



def make_dataset_random():
    path = '/cvlabdata1/cvlab/datasets_kyu/chest-xray14'
    # Open the h5file
    os.chdir('/cvlabdata1/cvlab/datasets_kyu/chest-xray14')

    h5file = h5py.File('Data_Entry_2017.h5')
    index_list = h5file['lists']
    train = index_list['train']
    valid = index_list['valid']
    test = index_list['test']

    entry = h5file['raw']['entry']
    nb_samples = entry.len()
    dis_dict = load_dict('category.json')
    indices = range(nb_samples)
    from random import shuffle
    nb_set = 5
    # for ind in range(nb_set):
    #     shuffle(indices)
    #     train[str(ind)] = sorted(indices[:split_loc[0]])
    #     valid[str(ind)] = sorted(indices[split_loc[0]: split_loc[1]])
    #     test[str(ind)] = sorted(indices[split_loc[1]:-1])

    if 'data_lists' not in h5file.keys():
        data_lists = h5file.create_group('data_lists')
    else:
        data_lists = h5file['data_lists']
    try:
        train_list = data_lists.create_group('train')
    except Exception as e: # TODO fix later
        train_list = data_lists['train']
    try:
        valid_list = data_lists.create_group('valid')
    except Exception as e:
        valid_list = data_lists['valid']
    try:
        test_list = data_lists.create_group('test')
    except Exception as e:
        test_list = data_lists['test']

    # Save the list to a json str.
    if 'json_label_lists' in h5file.keys():
        json_list = h5file['json_label_lists']
    else:
        json_list = h5file.create_group('json_label_lists')

    labels = ['train', 'valid', 'test']
    for l in labels:
        if l not in json_list.keys():
            json_list.create_group(l)

    for ind in range(nb_set):
        sind = str(ind)
        print("Create the index {}".format(sind))
        if sind not in train_list.keys():
            train_array = h5file['raw']['entry'][train[sind].value, :2]
            train_list[sind] = train_array
            # handle the json setting
            label_lists = [decode_diseases(item, dis_dict) for item in train_array[:, 1]]
            json_list['train'][sind] = json.dumps(label_lists)
        if sind not in valid_list.keys():
            valid_array = h5file['raw']['entry'][valid[sind].value, :2]
            valid_list[sind] = valid_array
            # handle the json setting
            label_lists = [decode_diseases(item, dis_dict) for item in valid_array[:, 1]]
            json_list['valid'][sind] = json.dumps(label_lists)
        if sind not in test_list.keys():
            test_array = h5file['raw']['entry'][test[sind].value, :2]
            test_list[sind] = test_array
            # handle the json setting
            label_lists = [decode_diseases(item, dis_dict) for item in test_array[:, 1]]
            json_list['test'][sind] = json.dumps(label_lists)

    h5file.close()


def decode_diseases(dis, dictionary):
    diseases = dis.split('|')
    if dis == 'No Finding':
        return []
    res = []
    for d in diseases:
        res.append(dictionary[d])
    return res


def read_csv_and_create_h5file():
    dataset_dir = '/Users/kcyu/data/chest-xray14'
    csvfile = 'Data_Entry_2017.csv'

    with open(os.path.join(dataset_dir, csvfile), 'rb') as f:
        reader = csv.reader(f)
        title = reader.next()
        entry = []
        for row in reader:
            entry.append(row)

    np_entry = np.asanyarray(entry)
    title = ['Image Index',
             'Finding Labels',
             'Follow-up',
             'Patient ID',
             'Patient Age',
             'Patient Gender',
             'View Position',
             'OriginalImage-Width',
             'OriginalImage-Height',
             'OriginalImagePixelSpacing-x',
             'OriginalImagePixelSpacing-y']

    h5file = h5py.File('Data_Entry_2017.h5','w')


class ChestXray14(ClassificationImageData):
    """ define the ChestXray pipeline """

    def __init__(self, dirpath='/home/kyu/.keras/datasets/chest-xray14',
                 image_dir='images', label_dir=None,
                 category='category.json',
                 config_file='Data_Entry_2017.h5',
                 single_label=None,
                 **kwargs):
        """

        Parameters
        ----------
        dirpath
        image_dir
        label_dir
        category
        config_file
        single_label : to become a binary classification problem!
        kwargs
        """
        super(ChestXray14, self).__init__(root_folder=dirpath, image_dir=image_dir,
                                          category=category, name='Chest-Xray-14',
                                          **kwargs)
        self.config_file = config_file
        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[256, 296], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=296, random_crop=False, horizontal_flip=True)
        self.single_label = single_label

    def _build_category_dict(self):
        # Load category
        if self.category_path.endswith('json'):
            self.category_dict = load_dict(self.category_path)
            self.nb_class = len(self.category_dict)
        else:
            raise NotImplementedError
        return self.category_dict

    def absolute_image_path(self, path):
        """
        Input a relative path and get a absolute
        :param path:
        :return:
        """
        return os.path.join(self.image_folder, path)

    def build_image_label_lists(self):
        # All logic goes here (including the shuffle)
        try:
            fdata = h5py.File(os.path.join(self.root_folder, self.config_file), 'r')
            # Load the train by index.
            train_list = fdata['data_lists']['train']
            valid_list = fdata['data_lists']['valid']
            test_list = fdata['data_lists']['test']
            json_list = fdata['json_label_lists']
            # Save to train list.
            for ind in range(5):
                sind = str(ind)
                self._set_train(
                    [self.absolute_image_path(p) for p in train_list[sind].value[:,0]],
                    json.loads(json_list['train'][sind].value),
                )
                self._set_valid(
                    [self.absolute_image_path(p) for p in valid_list[sind].value[:, 0]],
                    json.loads(json_list['valid'][sind].value),
                )
                self._set_test(
                    [self.absolute_image_path(p) for p in test_list[sind].value[:, 0]],
                    json.loads(json_list['test'][sind].value),
                )

        except IOError as e:
            raise NotImplementedError

    def generator(self, mode, indice, **kwargs):
        # mode = mode + str(indice)
        return super(ChestXray14, self).generator(mode, indice, class_mode='multi_categorical', **kwargs)


class ChestXray14Inference(ClassificationImageData):
    """ define the ChestXray pipeline """

    def __init__(self, dirpath='/home/kyu/.keras/datasets/chest-xray14',
                 image_dir='images', label_dir=None,
                 category='category.json',
                 config_file='Data_Entry_2017.h5',
                 boundingbox_file='BBox_List_2017.csv',
                 single_label=None,
                 image_data_generator=None,

                 **kwargs):
        """

        Parameters
        ----------
        dirpath
        image_dir
        label_dir
        category
        config_file
        single_label : to become a binary classification problem!
        kwargs
        """
        super(ChestXray14Inference, self).__init__(root_folder=dirpath, image_dir=image_dir,
                                                   category=category, name='Chest-Xray-14',
                                                   **kwargs)
        self.config_file = config_file
        self.boundingbox_file = boundingbox_file
        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[256, 296], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=296, random_crop=False, horizontal_flip=True)
        self.single_label = single_label

        # self.h5file = h5py.File(os.path.join(self.root_folder, self.config_file), 'r+')
        self.image_data_generator = ImageDataGeneratorAdvanced(target_size=(224,224), **self.valid_image_gen_configs ) \
            if image_data_generator is None else image_data_generator
        self.reset()

    def _build_category_dict(self):
        # Load category
        if self.category_path.endswith('json'):
            self.category_dict = load_dict(self.category_path)
            self.nb_class = len(self.category_dict)
        else:
            raise NotImplementedError
        self.category_dict['Infiltrate'] = 8
        return self.category_dict

    def absolute_image_path(self, path):
        """
        Input a relative path and get a absolute
        :param path:
        :return:
        """
        return os.path.join(self.image_folder, path)

    def build_image_label_lists(self):
        # For Inference, the output is defined as follow
        # It still output the target image, but with with different
        #
        try:
            fdata = h5py.File(os.path.join(self.root_folder, self.config_file), 'r+')
            # get the index
            self.id_bbindex = fdata['misc']['id_bbindex']
            self.boundingbox_entry = fdata['bb_raw']['entry']
            self.boundingbox_title = fdata['bb_raw']['title']
            self.image_entry = fdata['raw']['entry']
            self.id_entryindex = fdata['misc']['id_entryindex']

            # Get the image list
            self.image_list = [str(e)for e in self.id_bbindex]
            self.label_list = [self.image_entry[self.id_entryindex[e].value] for e in self.image_list]
            self.label_list = [l[1] for l in self.label_list]
            self.label_list = [decode_diseases(l, self.category_dict) for l in self.label_list]

            self.nb_sample = len(self.image_list)

        except IOError as e:
            raise NotImplementedError

    def reset(self):
        itr = Iterator(self.nb_sample, batch_size=1, shuffle=False, seed=0)
        self.index_generator = itr.index_generator

    def get_boundingbox(self, image_id):
        """
        Given the image ID (i.e., filename), generate the resulting bounding box.

        Parameters
        ----------
        image_id

        Returns
        -------

        """
        try:
            bb_list = [self.boundingbox_entry[e]for e in self.id_bbindex[image_id].value]
            img_entry = self.image_entry[self.id_entryindex[image_id].value]

            img_width ,img_height = float(img_entry[7]), float(img_entry[8])
            result = []
            labels = []
            for bb in bb_list:
                bbox = BoundingBox(center=(float(bb[2]), float(bb[3])),
                                   height=float(bb[5]),
                                   width=float(bb[4]),
                                   image_width=1024,
                                   image_height=1024,
                                   original_image_w=img_width,
                                   original_image_h=img_height)
                labels.append(bb[1])
                result.append(bbox)

        except ValueError as e:
            print(e)
            return None
        return result, labels

    def current_original(self, resize=False):
        index = self.current_index
        # load the corresponding test image.
        grayscale = False
        fname = self.absolute_image_path(self.image_list[index])
        img = load_img(fname, grayscale=grayscale)
        x = img_to_array(img, data_format=self.data_format)
        if resize:
            x = self.image_data_generator.advancedoperation(x)
        x = x.astype(K.floatx())
        return x,

    def current(self):
        index = self.current_index
        # load the corresponding test image.
        grayscale = False
        fname = self.absolute_image_path(self.image_list[index])
        img = load_img(fname, grayscale=grayscale)
        x = img_to_array(img, data_format=self.data_format)

        x = self.image_data_generator.advancedoperation(x)
        x = x.astype(K.floatx())
        x = self.image_data_generator.random_transform(x)
        x = self.image_data_generator.standardize(x)

        # load the label
        label = self.label_list[index]

        # load the corresponding bounding box
        bboxs, bb_labels = self.get_boundingbox(self.image_list[index])
        return x, label, bboxs, bb_labels

    def next(self):
        """
        Give the next data

        Returns
        -------
        Test Image: ndarray,
        Label :     ndarray,
        [BDBox] :     list [BoundBox,]

        """
        _, self.current_index, _ = self.index_generator.next()
        return self.current()

    def update_index_list(self, force=False):
        """
        Update the config.misc.id_entryindex, id_bbindex

            Give a Image ID, get the result index, for bb entry and boundbox
        Returns
        -------
        None
        """
        if not force:
            return
        # For entry
        config = self.h5file
        entry = config['raw']['entry']
        entryind_label = 'id_entryindex'
        if not entryind_label in config['misc'].keys() or force:
            del config['misc'][entryind_label]
            id_entryindex = config['misc'].create_group(entryind_label)
            for ind, e in enumerate(entry):
                id_entryindex[e[0]] = ind
        else:
            id_entryindex = config['misc'][entryind_label]

        # For bounding box entry
        entry = config['bb_raw']['entry']
        entryind_label = 'id_bbindex'
        if not entryind_label in config['misc'].keys() or force:
            del config['misc'][entryind_label]
            id_bbindex = config['misc'].create_group(entryind_label)

            bb_list = [e[0] for e in entry]
            unique_flist = list(set(bb_list))
            bb_file_dicts = dict()
            for ind, fname in enumerate(unique_flist):
                bb_file_dicts[fname] = []
            for ind, e in enumerate(entry):
                bb_file_dicts[e[0]].append(ind)

            # Generate the unique list
            for key, item in bb_file_dicts.items():
                id_bbindex[key] = item
        else:
            id_bbindex = config['misc'][entryind_label]

    def label_to_nnlabel(self, label):
        """

        Parameters
        ----------
        label

        Returns
        -------

        """
        nnlabel = np.zeros((self.nb_class,))
        for i in label:
            nnlabel[i] = 1
        return nnlabel


class ChestXray14SingleLabel(ClassificationImageData):
    """ define the ChestXray pipeline """

    def __init__(self,
                 single_label,
                 dirpath='/home/kyu/.keras/datasets/chest-xray14',
                 image_dir='images', label_dir=None,
                 category='category.json',
                 config_file='Data_Entry_2017.h5',
                 **kwargs):
        """

        Parameters
        ----------
        dirpath
        image_dir
        label_dir
        category
        config_file
        single_label : to become a binary classification problem!
        kwargs
        """

        # Set the single label
        if single_label is None or single_label > 14 or single_label < 0:
            raise ValueError("Only support int from 0 to 13")
        self.single_label = int(single_label)

        super(ChestXray14SingleLabel, self).__init__(root_folder=dirpath, image_dir=image_dir,
                                                     category=category,
                                                     name='Chest-Xray-14-label_{}'.format(single_label),
                                                     **kwargs)

        self.config_file = config_file
        self.build_image_label_lists()
        self.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[225, 256], random_crop=True, horizontal_flip=True)
        self.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=256, random_crop=False, horizontal_flip=True)

    def _build_category_dict(self):
        # Load category
        if self.category_path.endswith('json'):
            self.category_dict = load_dict(self.category_path)
            self.nb_class = len(self.category_dict)
        else:
            raise NotImplementedError
        # Modify for the single label, change the dictionary.
        target_key = None
        self.label_dict = dict()
        for key, value in self.category_dict.items():
            if value != self.single_label:
                self.label_dict[value] = 0
                self.category_dict[key] = 0
            else:
                target_key = key
                self.label_dict[value] = 1
                self.category_dict[key] = 1
        self.nb_class = 2
        return self.category_dict

    def absolute_image_path(self, path):
        """
        Input a relative path and get a absolute
        :param path:
        :return:
        """
        return os.path.join(self.image_folder, path)

    def build_image_label_lists(self):
        # All logic goes here (including the shuffle)
        try:
            fdata = h5py.File(os.path.join(self.root_folder, self.config_file), 'r')
            # Load the train by index.
            train_list = fdata['data_lists']['train']
            valid_list = fdata['data_lists']['valid']
            test_list = fdata['data_lists']['test']
            json_list = fdata['json_label_lists']
            # Save to train list.
            for ind in range(5):
                sind = str(ind)
                self._set_train(
                    [self.absolute_image_path(p) for p in train_list[sind].value[:,0]],
                    json.loads(json_list['train'][sind].value),
                )
                self._set_valid(
                    [self.absolute_image_path(p) for p in valid_list[sind].value[:, 0]],
                    json.loads(json_list['valid'][sind].value),
                )
                self._set_test(
                    [self.absolute_image_path(p) for p in test_list[sind].value[:, 0]],
                    json.loads(json_list['test'][sind].value),
                )

        except IOError as e:
            raise NotImplementedError

    def generator(self, mode, indice, **kwargs):
        # mode = mode + str(indice)
        gen = super(ChestXray14SingleLabel, self).generator(mode, indice, class_mode='binary', **kwargs)
        gen.category_dict = self.label_dict
        return gen



if __name__ == '__main__':
    # make_dataset_random()
    # gen = ImageDataGeneratorAdvanced()
    inference = ChestXray14Inference()
    x, label, bbxs = inference.next()
