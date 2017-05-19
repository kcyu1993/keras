"""
Load for xray data

"""
from scipy.misc import imresize

import keras.backend as K
import json
import os
import numpy as np
from imagenet import preprocess_image_for_imagenet, ImageIterator


def preprocess(img, crop=True, resize=True, dsize=(224, 224)):
    """
    Ref: Xvision github.
    
    Parameters
    ----------
    img
    crop
    resize
    dsize

    Returns
    -------

    """
    if img.dtype == np.uint8:
        img = img / 255.0

    if crop:
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    else:
        crop_img = img

    if resize:
        norm_img = imresize(crop_img, dsize, preserve_range=True)
    else:
        norm_img = crop_img

    return (norm_img).astype(np.float32)


def deprocess(img):
    """
    References: Xvision github
    
    Parameters
    ----------
    img

    Returns
    -------

    """
    return np.clip(img * 255, 0, 255).astype(np.uint8)
    # return ((img / np.max(np.abs(img))) * 127.5+127.5).astype(np.uint8)


def load_image_label_from_json(path):
    """
    Load the Json file and return the pair
    Parameters
    ----------
    path: path to json

    Returns
    -------
    filename list, category list
    """

    with open(path) as data_file:
        data = json.load(data_file)
    keys = [int(k) for k in data.keys()]
    values = [0 if v == 'normal' else 1 for v in data.values()]
    return keys, values


class XrayLoader(object):
    """
    Define the Xray data loader.
    Create for Swisscom testing project.
    
    it reads the current directory structure
    
    data_path/
        xxx.png
    
    xxx_test.json
    xxx_traing.json
    
    
    Special rule:
        for image with weight > height, square centered then shrink.
        for image with weight < height, square centered then shrink.
        
    
    """

    def __init__(self, data_path='/home/kyu/cvkyu/swisscom/data',
                 metadata_path='/home/kyu/cvkyu/swisscom/',
                 dim_ordering='default'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.dim_ordering = dim_ordering

        self.nb_class = 2 # binary classification

        self.data_path = data_path
        self.metadata_path = metadata_path

    def generator(self, mode='train', batch_size=32, target_size=(224,224),
                  image_data_generator=None, **kwargs):
        """
        get the generator for training and testing.
        
        Parameters
        ----------
        mode
        batch_size
        target_size
        image_data_generator
        kwargs

        Returns
        -------

        """
        if mode not in {'train', 'test'}:
            raise ValueError("Mode should be either train or test")
        if mode == 'train':
            metadata = self.metadata_path + 'normalVSrest_trainingSet.json'
        else:
            metadata = self.metadata_path + 'normalVSrest_testSet.json'

        flist, cate_list = load_image_label_from_json(metadata)
        cate_list = np.asanyarray(cate_list)

        # Assign the flist
        flist = [os.path.join(self.data_path, str(f) + '.png') for f in flist]

        generator = ImageIterator(flist, cate_list, self.nb_class,
                                  image_data_generator=image_data_generator,
                                  batch_size=batch_size, target_size=target_size,
                                  dim_ordering=self.dim_ordering, **kwargs
                                  )
        return generator
