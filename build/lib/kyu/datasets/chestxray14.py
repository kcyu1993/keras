"""
Implement the loading for Chest Xray dataset.
Currently for classification tasks

"""
import csv
import os
import numpy as np
import h5py
import sys
from kyu.engine.utils.data_utils import ClassificationImageData
from kyu.utils.dict_utils import create_dict_by_given_kwargs, load_dict
import json
import keras.backend as K


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

            # return title, entry


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


if __name__ == '__main__':
    make_dataset_random()
