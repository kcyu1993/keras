"""
Cleaning and re-implemtation of CIFAR experiments, with the brand new model design

"""
from keras.applications.resnet50 import ResNet50CIFAR
from keras.layers import O2Transform, SecondaryStatistic

from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.optimizers import SGD
from keras.utils import np_utils

import keras.backend as K
from keras.utils.data_utils import get_absolute_dir_project

from kyu.models.fitnet import fitnet_o2, fitnet_o1
from kyu.models.resnet import ResNet50_cifar_o2
from kyu.theano.cifar.configs import get_von_with_regroup, get_experiment_settings, get_aaai_experiment, get_matrix_bp, \
    get_cov_beta_cv, get_resnet_batch_norm, get_resnet_with_power, get_resnet_with_bn, get_resnet_experiments, \
    get_residual_cov_experiment
from kyu.theano.general.finetune import get_tmp_weights_path, run_finetune, run_finetune_with_Stiefel_layer, \
    run_finetune_with_weight_norm
from kyu.theano.general.model import get_so_model_from_config
from kyu.theano.general.train import fit_model_v2
from kyu.utils.train_utils import toggle_trainable_layers
from third_party.openai.weightnorm import SGDWithWeightnorm

if K._BACKEND == 'tensorflow':
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')

batch_size = 32
nb_classes = 10
nb_epoch = 10
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

input_shape = (32, 32, 3)
BASELINE_PATH = get_absolute_dir_project('model_saved/cifar10_baseline.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_cnn_sndstat.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_fitnet.weights')
LOG_PATH = get_absolute_dir_project('model_saved/log')

def cifar10_data():

    cifar_10 = True
    label_mode = 'fine'
    if cifar_10:
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        nb_classes = 10
    else:
        # the data, shuffled and split between train and test sets
        # label_mode = 'fine'
        (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
        if label_mode is 'fine':
            print('use cifar 100 fine')
            nb_classes = 100
        elif label_mode is 'coarse':
            print('use cifar 100 coarse')
            nb_classes = 20

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # input_shape = X_train.shape[1:]
    return [X_train, Y_train], [X_test, Y_test]


def cifar_train(model,
                nb_epoch_finetune=100, nb_epoch_after=0, batch_size=32,
                image_gen=None,
                title='cifar10_train', early_stop=True,
                keyword='',
                optimizer=None,
                log=True,
                verbose=2,
                lr_decay=False,
                lr=0.01):

    train, test = cifar10_data()

    if optimizer is None:
        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    fit_model_v2(model, [train, test], batch_size=batch_size, title=title,
                 nb_epoch=nb_epoch_finetune,
                 optimizer=optimizer,
                 early_stop=early_stop,
                 verbose=verbose,
                 log=log,
                 lr=lr,
                 lr_decay=lr_decay)
    tmp_weights = get_tmp_weights_path(model.name)
    model.save_weights(tmp_weights)
    if nb_epoch_after > 0:
        if nb_epoch_finetune > 0:
            lr /= 10
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
                     lr=lr)
        model.save_weights()


def run_model_with_config(model, config, title='cifar10',
                          image_gen=None,
                          verbose=(2, 2), nb_epoch_finetune=15, nb_epoch_after=50,
                          stiefel_observed=None, stiefel_lr=0.01,
                          weight_norm=False,
                          lr_decay=False):
    """
    Finetune the ResNet-DCov

    Returns
    -------

    """

    monitor_class = (O2Transform, SecondaryStatistic)
    # monitor_metrics = ['weight_norm',]
    # monitor_metrics = ['output_norm',]
    monitor_metrics = ['matrix_image',]
    if stiefel_observed is None:
        if weight_norm:
            run_finetune_with_weight_norm(
                model, cifar_train,
                nb_classes=nb_classes,
                input_shape=input_shape, config=config,
                nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                image_gen=image_gen, title=title + '-weight_norm', verbose=verbose,
                monitor_classes=monitor_class,
                monitor_measures=monitor_metrics,
                lr_decay=lr_decay
            )
        run_finetune(model, cifar_train,
                     nb_classes=nb_classes,
                     input_shape=input_shape, config=config,
                     nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                     image_gen=image_gen, title=title, verbose=verbose,
                     monitor_classes=monitor_class,
                     monitor_measures=monitor_metrics,
                     lr_decay=lr_decay)
    else:
        run_finetune_with_Stiefel_layer(model, cifar_train,
                                        nb_classes=nb_classes,
                                        input_shape=input_shape, config=config,
                                        nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after,
                                        image_gen=image_gen, title=title + "-stiefel", verbose=verbose,
                                        monitor_classes=monitor_class,
                                        monitor_measures=monitor_metrics,
                                        observed_keywords=stiefel_observed,
                                        lr=stiefel_lr,
                                        lr_decay=lr_decay)


def run_routine_fitnet(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                       stiefel_observed=None, stiefel_lr=0.01,
                       lr_decay=False):
    run_model_with_config(fitnet_o2, config, title='cifar_fitnet',
                          verbose=verbose, image_gen=None,
                          nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                          stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed,
                          lr_decay=lr_decay,
                          )


def run_routine_resnet(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,
                       stiefel_observed=None, stiefel_lr=0.01,
                       weight_norm=False,
                       lr_decay=False):
    run_model_with_config(ResNet50_cifar_o2, config, title='cifar_resnet',
                          verbose=verbose, image_gen=None,
                          nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                          stiefel_lr=stiefel_lr, stiefel_observed=stiefel_observed,
                          lr_decay=lr_decay,
                          weight_norm=weight_norm
                          )

def run_routine_resnet_weight_norm(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50,lr_decay=False):

    model = get_so_model_from_config(ResNet50_cifar_o2, config.params[0], config.mode_list[0],config.cov_outputs[0],
                                     nb_classes, input_shape=input_shape, config=config)
    gsgd_0 = SGDWithWeightnorm(0.01, 0.2, 0, False)
    cifar_train(model, nb_epoch_finetune=nb_epoch_finetune, nb_epoch_after=nb_epoch_after, batch_size=config.batch_size,
                title='cifar10_resnet_weight_norm' + config.title, early_stop=config.early_stop,
                optimizer=gsgd_0,
                )


def baseline_resnet(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50, batch_norm=True):

    # Batch norm is disable
    model = ResNet50CIFAR(input_shape=input_shape, nb_class=nb_classes, batch_norm=batch_norm)

    cifar_train(model, nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                batch_size=config.batch_size, verbose=verbose[0], early_stop=False, lr_decay=False,
                title='cifar10_resnet_v1_baseline_no_batchnorm')


def baseline_fitnet(config, verbose=(2,2), nb_epoch_finetune=15, nb_epoch_after=50):

    model = fitnet_o1(denses=[500], nb_classes=nb_classes, input_shape=input_shape, version=1)

    cifar_train(model, nb_epoch_after=nb_epoch_after, nb_epoch_finetune=nb_epoch_finetune,
                batch_size=config.batch_size, verbose=verbose[0], early_stop=True, lr_decay=False,
                title='cifar10_fitnet_v1_baseline')


if __name__ == '__main__':
    exp = 2
    weight_norm = True
    # config = get_resnet_experiments(1)
    config = get_residual_cov_experiment(1)
    # config = get_resnet_with_power(exp)
    # config = get_resnet_with_bn(exp)
    # config = get_resnet_batch_norm(exp)
    # config = get_cov_beta_cv(1)
    # config = get_experiment_settings(9)
    # config.cov_mode = 'channel'
    # config = get_experiment_settings(2)
    # config = get_aaai_experiment(1)
    # config = get_matrix_bp(1)
    # config = get_von_with_regroup(1)
    # run_routine_fitnet(config, nb_epoch_after=100, nb_epoch_finetune=0,
    #                    verbose=(2,1),
    #                    stiefel_lr=(0.01,0.001), stiefel_observed=['o2t']
                       # )
    # baseline_fitnet(config, verbose=(2,2), nb_epoch_finetune=0, nb_epoch_after=200)
    # run_routine_fitnet(config, nb_epoch_after=200, nb_epoch_finetune=0, lr_decay=False)
    run_routine_resnet_weight_norm(config, nb_epoch_after=200, nb_epoch_finetune=0, lr_decay=False)
    # run_routine_resnet(config, nb_epoch_after=200, nb_epoch_finetune=0, lr_decay=True,
    #                    weight_norm=weight_norm,
                       # stiefel_observed=['o2t'], stiefel_lr=(0.01,0.001)
                       # )
    # baseline_resnet(config, nb_epoch_after=200, nb_epoch_finetune=0, batch_norm=False)
