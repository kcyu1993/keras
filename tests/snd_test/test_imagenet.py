import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pytest


from keras.utils.test_utils import keras_test
from keras import backend as K
from keras.applications.vgg16 import VGG16

from kyu.datasets.imagenet import ImageNetTools, ImageNetLoader

PATH = '/home/kyu/.keras/datasets/ILSVRC2015'
IMAGENET_VALID_GROUNDTRUTH_FILE = 'ILSVRC2014_clsloc_validation_ground_truth.txt'
IMAGENET_VALID_BLACKLIST_FILE = 'ILSVRC2014_clsloc_validation_blacklist.txt'

PLOT_PATH = '/cvlabdata1/home/kyu/plots/imagenettest'

def get_labels(input_list):
    tool = ImageNetTools(os.path.join(PATH, 'ImageSets/CLS-LOC/meta_clsloc.mat'))
    return [tool.id_to_words((int(v))) for v in input_list]


def translate_result_from_VGG(input_list):
    tool = ImageNetTools(os.path.join(PATH, 'ImageSets/CLS-LOC/meta_clsloc.mat'))
    input_list = input_list.tolist()
    return [tool.synset_to_id(tool.synsets_imagenet_sorted[int(i)][1]) for i in input_list]

def test_imagenet_tool():
    tool = ImageNetTools(os.path.join(PATH, 'ImageSets/CLS-LOC/meta_clsloc.mat'))
    f = open(os.path.join(PATH, 'ImageSets/CLS-LOC', IMAGENET_VALID_GROUNDTRUTH_FILE), 'r')
    lines = [v for v in f.readlines()]
    lines = lines[:10]

    result = [tool.id_to_words(int(v)) for v in lines]
    for r in result:
        print(r)


def test_imagenet_loader():
    loader = ImageNetLoader(PATH)
    l = loader.generator(mode='valid', image_data_generator=None,
                         batch_size=4,
                         save_to_dir=PLOT_PATH, save_prefix='test_loader',
                         shuffle=False)
    ba, bb = l.next()
    res = get_labels(bb.argmax(1) + 1)
    for r in res:
        print(r)


def test_imagenet_model_VGG():
    batch_size = 16
    loader = ImageNetLoader(PATH)
    l = loader.generator(mode='valid', image_data_generator=None,
                         batch_size=batch_size,
                         save_to_dir=PLOT_PATH, save_prefix='test_loader',
                         shuffle=False)
    ba, bb = l.next()
    res = get_labels(bb.argmax(1) + 1)
    model = VGG16()
    model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['accuracy'])
    imgmean = [ 103.939, 116.779, 123.68]
    test_img  = ba - imgmean
    score = model.evaluate(test_img, bb, batch_size=batch_size)
    pred = model.predict(test_img, batch_size=batch_size)
    pred_label = pred.argmax(1)

    # res2 = get_labels(pred_label + 1)
    # print("pred label " , pred_label)
    # print("Predicted result vs original result: ")
    # for r1, r2 in zip(res2, res):
    #     print(r1 + ' == '  + r2)

    print("translate result")
    real_label_t = translate_result_from_VGG(bb.argmax(1))
    pred_label_t = translate_result_from_VGG(pred_label)
    print("pred label ", pred_label_t)
    res_t = get_labels(np.array(pred_label_t))
    res = get_labels(np.array(real_label_t))
    print("Predicted result vs original result: ")
    for r1, r2 in zip(res_t, res):
        print(r1 + ' == ' + r2)

    print('score', score)

if __name__ == '__main__':
    # test_imagenet_tool()
    # test_imagenet_loader()

    test_imagenet_model_VGG()
