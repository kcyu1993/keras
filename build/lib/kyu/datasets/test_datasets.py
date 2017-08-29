from kyu.datasets.dtd import DTD
from kyu.datasets.minc import Minc2500_v2, TmpMinc

from kyu.engine.utils.data_utils import ImageData, ClassificationImageData


def test_dataset_validity(data, data_class):
    print("Asserting {} for ClassificationImageData subclass".format(data_class))
    print(issubclass(data_class, ClassificationImageData))
    print("Asserting {} for ImageData subclass".format(data_class))
    print(issubclass(data_class, ImageData))

    print("Asserting {} for ClassificationImageData instance".format(data))
    print(isinstance(data, ClassificationImageData))
    print("Asserting {} for ImageData instance".format(data))
    print(isinstance(data, ImageData))


def test_tmp_minc():
    minc = TmpMinc(root_folder='/home/kyu/.keras/datasets/minc-2500', image_dir=None)
    test_dataset_validity(minc, TmpMinc)


def test_minc():
    minc = Minc2500_v2(dirpath='/home/kyu/.keras/datasets/minc-2500', image_dir=None)
    test_dataset_validity(minc, Minc2500_v2)


def test_dtd():
    dtd = DTD('/home/kyu/.keras/datasets/dtd')
    test_dataset_validity(dtd, DTD)


if __name__ == '__main__':
    test_dtd()
    test_minc()
    test_tmp_minc()