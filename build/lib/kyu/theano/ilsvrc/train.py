"""
Train with ImageNet dataset

"""

import os

import keras.backend as K
from keras.preprocessing.image import ImageDataGeneratorAdvanced
from kyu.datasets.imagenet import preprocess_image_for_imagenet, ImageNetLoader

# Some constants
from kyu.models.resnet import ResNet50_o2_multibranch
from kyu.models.vgg import VGG16_o2
from kyu.theano.general.finetune import finetune_model_with_config
from kyu.theano.general.train import fit_model_v2, toggle_trainable_layers
from kyu.theano.ilsvrc.config import get_VGG_testing_ideas

nb_classes = 1000
if K.backend() == 'tensorflow':
    input_shape=(224,224,3)
    K.set_image_dim_ordering('tf')
else:
    input_shape=(3,224,224)
    K.set_image_dim_ordering('th')

TARGET_SIZE = (224,224)
RESCALE_SMALL = 256

NB_EPOCH = 1000
VERBOSE = 1
SAVE_LOG = True
VALIDATION = True
BATCH_SIZE = 32


def get_tmp_weights_path(name):
    return '/tmp/{}_finetune.weights'.format(name)


def loadImageNet(image_gen=None,batch_size=32, ):
    # Absolute paths
    IMAGENET_PATH = '/home/kyu/.keras/datasets/ILSVRC2015'
    TARGET_SIZE = (224, 224)
    RESCALE_SMALL = 256
    # ImageNet generator
    imageNetLoader = ImageNetLoader(IMAGENET_PATH)
    if image_gen is None:
        iamge_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                               horizontal_flip=True,
                                               preprocessing_function=preprocess_image_for_imagenet
                                               # channelwise_std_normalization=True
                                               )

    train = imageNetLoader.generator('train', image_data_generator=image_gen, batch_size=batch_size)
    valid = imageNetLoader.generator('valid', image_data_generator=image_gen, batch_size=batch_size)
    # test = imageNetLoader.generator('valid', image_data_generator=gen)
    return train, valid


def imagenet_finetune(model,
                      nb_epoch_finetune=100, nb_epoch_after=0, batch_size=32,
                      image_gen=None,
                      title='ImageNet_finetune', early_stop=False,
                      keyword='',
                      optimizer=None,
                      log=True,
                      lr_decay=True,
                      verbose=2,
                      lr=0.001):

    train, test = loadImageNet()

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    fit_model_v2(model, [train, test], batch_size=batch_size, title=title,
                 nb_epoch=nb_epoch_finetune,
                 optimizer=optimizer,
                 early_stop=early_stop,
                 verbose=verbose,
                 lr_decay=lr_decay,
                 log=log,
                 lr=lr)
    tmp_weights = get_tmp_weights_path(model.name)
    model.save_weights(tmp_weights)
    if nb_epoch_after > 0:
        # K.clear_session()
        toggle_trainable_layers(model, True, keyword)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.load_weights(tmp_weights)
        fit_model_v2(model, [train, test], batch_size=batch_size, title=title,
                     nb_epoch=nb_epoch_after,
                     optimizer=optimizer,
                     early_stop=early_stop,
                     verbose=verbose,
                     lr_decay=lr_decay,
                     lr=lr/10)

    return


def run_routine_vgg(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                    stiefel_observed=None, stiefel_lr=0.01):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           preprocessing_function=preprocess_image_for_imagenet
                                           )

    # monitor_class = (O2Transform, SecondaryStatistic)
    # monitor_metrics = ['weight_norm',]
    # monitor_metrics = ['output_norm',]
    # monitor_metrics = ['matrix_image',]
    finetune_model_with_config(VGG16_o2, imagenet_finetune, config, nb_classes,
                               input_shape=input_shape,
                               title='ImageNet_VGG16',
                               verbose=verbose, image_gen=image_gen,
                               nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                               stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed)


def run_routine_resnet_multibranch(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                                   stiefel_observed=None, stiefel_lr=0.01):
    """
        Finetune the ResNet-DCov

        Returns
        -------

        """
    image_gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                           horizontal_flip=True,
                                           )

    finetune_model_with_config(ResNet50_o2_multibranch, imagenet_finetune, config, nb_classes,
                               title='ImageNet_ResNet_MB',
                               verbose=verbose, image_gen=image_gen,
                               nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                               stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed)


if __name__ == '__main__':
    exp = 1
    config = get_VGG_testing_ideas(exp)
    run_routine_vgg(config, verbose=(2,2), nb_epoch_finetune=3, nb_epoch_after=NB_EPOCH)
