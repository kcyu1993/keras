from kyu.configs.dataset_config import DatasetConfig

from airplane import Aircraft
from chestxray14 import ChestXray14
from cub import CUB
from dtd import DTD
from imagenet import ImageNetData as ImageNet
from kyu.datasets.food101 import Food101
from kyu.datasets.stanford_car import StanfordCar
from minc import Minc2500_v2 as Minc2500
from mit import MitIndoor
from sun import SUN397_v2 as SUN397
# from common_imports import *


def get_dataset_by_name(name, dirpath=None):
    """
    Get dataset by the name

    Parameters
    ----------
    name
    dirpath

    Returns
    -------

    """
    config = DatasetConfig(name, dirpath=dirpath)
    return get_dataset(config)


def get_dataset(config):
    """
    Define the dataset config

    Parameters
    ----------
    config

    Returns
    -------

    """
    if not isinstance(config, DatasetConfig):
        raise ValueError("Get dataset must be a DatasetConfig file")

    identifier = str(config.name).lower()
    if identifier in ['dtd']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/dtd'
        dataset = DTD(config.dirpath)
    elif identifier in ['minc2500', 'minc-2500']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/minc-2500'
        dataset = Minc2500(config.dirpath)
    elif identifier in ['mit', 'mit67', 'mitinddor', 'mit-indoor']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/mit_indoor'
        dataset = MitIndoor(config.dirpath)
    elif identifier in ['sun', 'sun397']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/sun'
        dataset = SUN397(config.dirpath)
    elif identifier in ['imagenet', 'ilsvrc']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/ILSVRC2015'
        dataset = ImageNet(config.dirpath)
    elif identifier in ['chestxray', 'chest-xray', 'chest-xray14']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/chest-xray14'
        dataset = ChestXray14(config.dirpath)
    elif identifier in ['cub', 'cub2001', 'cub200']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/cub/CUB_200_2011'
        dataset = CUB(config.dirpath)
    elif identifier in ['airplane', 'aircraft']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/fgvc-aircraft-2013b/data'
        dataset = Aircraft(config.dirpath)
    elif identifier in ['car', 'stanford_car']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/car'
        dataset = StanfordCar(config.dirpath)
    elif identifier in ['food', 'food101', 'food-101']:
        config.dirpath = config.dirpath if config.dirpath is not None \
            else '/home/kyu/.keras/datasets/food-101/food-101'
        dataset = Food101(config.dirpath)
    else:
        raise ValueError("Dataset Name Not recognized {}".format(identifier))

    return dataset




