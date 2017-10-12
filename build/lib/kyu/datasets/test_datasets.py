from kyu.datasets.dtd import DTD
from kyu.datasets.minc import Minc2500_v2, TmpMinc
from kyu.datasets.chestxray14 import ChestXray14
from kyu.engine.utils.data_utils import ImageData, ClassificationImageData
from kyu.utils.image import ImageDataGeneratorAdvanced


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


def test_chestxray():
    data = ChestXray14()
    TARGET_SIZE = (224, 224)
    RESCALE_SMALL = (256, 296)

    gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                     horizontal_flip=True,
                                     )

    valid_gen = ImageDataGeneratorAdvanced(TARGET_SIZE,
                                           rescaleshortedgeto=256,
                                           random_crop=False,
                                           horizontal_flip=True)

    valid = data.get_test(image_data_generator=valid_gen, save_to_dir='/home/kyu/plots',
                          save_prefix='chest_valid', save_format='JPEG',
                          shuffle=False)

    train = data.get_train(image_data_generator=gen, save_to_dir='/home/kyu/plots',
                           save_prefix='chest', save_format='JPEG',
                           shuffle=False)
    # a, b = train.next()
    a, b = valid.next()
    a, b = train.next()


if __name__ == '__main__':
    # test_dtd()
    # test_minc()
    # test_tmp_minc()
    test_chestxray()