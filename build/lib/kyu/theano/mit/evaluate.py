"""
Evaluate the given model based on weights
"""
import time

from kyu.theano.minc.configs import get_von_with_regroup, get_VGG_testing_ideas, get_ResNet_testing_ideas

from keras.applications import ResNet50

from kyu.datasets.minc import MincOriginal, ImageDataGeneratorAdvanced, Minc2500, load_minc2500
import keras.backend as K
import os

# Some constants
from kyu.models.resnet import ResNet50_o1, ResNet50_o2, ResNet50_o2_with_config
from kyu.models.vgg import VGG16_o2, VGG16_o2_with_config, VGG16_o1
from kyu.theano.general.config import DCovConfig

nb_classes = 23
if K.backend() == 'tensorflow':
    input_shape=(224,224,3)
    K.set_image_dim_ordering('tf')
else:
    input_shape=(3,224,224)
    K.set_image_dim_ordering('th')

TARGET_SIZE = (224,224)
RESCALE_SMALL = 256


WEIGHTS_ROOT = '/home/kyu/.keras/models/best/minc2500'
WEIGHTS_FOLDER = '/home/kyu/.keras/models'


def load_minc_original(filename='test.txt', batch_size=32, gen=None, save_dir=None):
    loader = MincOriginal()
    test_gen = loader.generator(filename, batch_size=batch_size, shuffle=True, gen=gen, target_size=TARGET_SIZE,
                                save_dir=save_dir)
    return test_gen


def minc_orig_evaluate(model,
                       batch_size=32,
                       image_gen=None,
                       title='evaluate',
                       verbose=1,
                       ):
    if image_gen is None:
        image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                               horizontal_flip=True,
                                               )
    test = load_minc_original(batch_size=batch_size, gen=image_gen)
    result = model.evaluate_generator(test, 1000, nb_worker=10,
    # result = model.evaluate_generator(test, test.nb_sample, nb_worker=10,
                                      max_q_size=20)
    print(title)
    print(result)


def minc2500_evaluate(model,
                      batch_size=32,
                      image_gen=None,
                      title='evaluate',
                      verbose=1,
                      ):
    if image_gen is None:
        image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                               horizontal_flip=True,
                                               )
    loader = Minc2500()
    train, test = load_minc2500(index=1, target_size=TARGET_SIZE, gen=image_gen, batch_size=batch_size)
    stime = time.time()
    result = model.evaluate_generator(test, 1000, nb_worker=10,
    # result = model.evaluate_generator(test, test.nb_sample, nb_worker=10,
                                      max_q_size=20)
    print("--- %s seconds ---" % (time.time() - stime))
    print(title)
    print(result)





def minc_evaluate_resnet_baseline():
    model = ResNet50_o1(denses=[], nb_classes=nb_classes, input_shape=input_shape)
    model.load_weights(os.path.join(WEIGHTS_ROOT, 'minc2500_finetune_resnet-baseline_resnet_1.weights'), by_name=True)
    model.compile('sgd', loss='categorical_crossentropy', metrics=['acc'])

    # minc_orig_evaluate(model, batch_size=32, title='baseline_evaluate')
    minc2500_evaluate(model, batch_size=32, title='baseline_evaluate')


def minc_evaluate_resnet_o2_best():
    weights_path = 'retrain_minc2500_resnet50minc_von_mean_o2transform_robost_None_cov_o2transform_' \
                   'wv64_mean-ResNet_o2_o2transform_para-257_128_64_mode_1nb_branch_4_1.weights.tmp'

    weights_path = os.path.join(WEIGHTS_ROOT, weights_path)
    # config = get_von_with_regroup(2)
    config = get_ResNet_testing_ideas(2)
    model = ResNet50_o2_with_config(config.params[0], config.mode_list[0], config.cov_outputs[0], config,
                                    nb_class=23, input_shape=(224,224,3))
    minc_evaluate_model_with_weights(model, weights_path)


def minc_evaluate_resnet_any():
    """
    Any ResNet model examine.

    Returns
    -------

    """
    resnet_stiefel_br_4 = 'finetune_minc2500_resnet50_stiefelminc_von_mean_o2transform_robost_None_cov_' \
                          'o2transform_wv64_pmean-ResNet_o2_o2transform_para-513_257_128_64_mode_1nb_branch_4_1.weights'
    resnet_stiefel_br_4 = os.path.join(WEIGHTS_FOLDER, resnet_stiefel_br_4)
    weights_path = resnet_stiefel_br_4
    config = get_von_with_regroup(1)
    model = ResNet50_o2_with_config(config.params[0], config.mode_list[0], config.cov_outputs[0], config,
                                    nb_class=23, input_shape=(224, 224, 3))
    minc_evaluate_model_with_weights(model, weights_path)




def minc_evaluate_vgg_baseline():
    model = VGG16_o1(denses=[4096, 4096, 4096], nb_classes=nb_classes, input_shape=input_shape)
    # model.load_weights(os.path.join(WEIGHTS_ROOT, 'minc2500_finetune_resnet-baseline_resnet_1.weights'), by_name=True)
    model.compile('sgd', loss='categorical_crossentropy', metrics=['acc'])

    # minc_orig_evaluate(model, batch_size=32, title='baseline_evaluate_vgg')
    minc2500_evaluate(model, batch_size=32, title='baseline_evaluate_vgg')



def minc_evaluate_vgg_o2_best():
    """
    Evaluate the best VGG model, actually it is same to run it.

    Returns
    -------

    """
    vgg_weight = ''
    vgg_weight = os.path.join(WEIGHTS_ROOT, vgg_weight)
    config = get_VGG_testing_ideas(1)
    model = VGG16_o2_with_config(config.params[0], config.mode_list[0], config.cov_outputs[0], config,
                                 nb_class=23, input_shape=(224, 224, 3))
    minc_evaluate_model_with_weights(model, weights_path=None)


def examine_weights(weights_path):
    config = get_von_with_regroup(2)

    model = ResNet50_o1(nb_classes=23, input_shape=input_shape, load_weights=False)
    model.load_weights(weights_path, True)


def minc_evaluate_model_with_weights(model, weights_path):
    model.compile('sgd', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    if weights_path:
        model.load_weights(weights_path, True)
    minc2500_evaluate(model, batch_size=32, title='baseline_evaluate')
    # minc_orig_evaluate(model, batch_size=32, title='baseline_evaluate')


if __name__ == '__main__':
    # minc_evaluate_resnet_baseline()
    # vgg_stiefel_weights_path = 'finetune_minc2500_resnet50_stiefelminc_von_mean_o2transform_robost_None_' \
    #                            'cov_o2transform_wv64_pmean-VGG16_o2_para-257_128_64_mode_1nb_branch_2_1.weights'
    minc_evaluate_resnet_o2_best()
    # examine_weights(os.path.join(WEIGHTS_FOLDER, vgg_stiefel_weights_path))

    # minc_evaluate_resnet_any()

    # minc_evaluate_vgg_o2_best()
    # minc_evaluate_vgg_baseline()