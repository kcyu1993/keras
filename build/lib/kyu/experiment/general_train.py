import argparse

from keras.layers import Conv2D, BatchNormalization
from keras.optimizers import SGD
from kyu.engine.trainer import ClassificationTrainer
from kyu.models import get_model
from kyu.utils.image import get_vgg_image_gen, get_resnet_image_gen, get_densenet_image_gen
from kyu.utils.io_utils import ProjectFile


def get_dirhelper(dataset_name, model_category, **kwargs):
    return ProjectFile(root_path='/home/kyu/cvkyu/so_updated_record', dataset=dataset_name, model_category=model_category,
                       **kwargs)

def get_debug_dirhelper(dataset_name, model_category, **kwargs):
    return ProjectFile(root_path='/home/kyu/cvkyu/debug_secondstat', dataset=dataset_name, model_category=model_category,
                       **kwargs)


def get_argparser(description='default'):
    """
    Define the get argument parser default, given the description of the task

    Parameters
    ----------
    description: str  description of the task.

    Returns
    -------
    parser with all defaults

    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset name: support dtd, minc2500')
    parser.add_argument('-me', '--model_exp', help='model experiment index', type=int, default=1)
    parser.add_argument('-m', '--model_class', help='model class should be in vgg, resnet', default='vgg', type=str)
    parser.add_argument('-ef', '--nb_epoch_finetune', help='number of epoch to finetune', default=0, type=int)
    parser.add_argument('-et', '--nb_epoch_train', help='number of epoch to retrain', default=200, type=int)
    parser.add_argument('-c', '--comments', help='comments if any', default='', type=str)
    parser.add_argument('-lr', '--learning_rate', help='learning rate initial', default=0.01, type=float)
    parser.add_argument('--channel_reverse', help='enable channel transform from RGB to BGR', default=False, type=bool)
    parser.add_argument('-init_weight', '--init_weights_location', help='init weights location', default='', type=str)

    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='Enable Tensorboard monitoring', dest='tensorboard')
    parser.add_argument('-ntb', '--no-tensorboard', action='store_false',
                        help='Disable tensorboard monitoring', dest='tensorboard',)
    parser.set_defaults(tensorboard=False)

    parser.add_argument('-debug', '--debug', help='set debug flag', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    parser.add_argument('-spe', '--save_per_epoch', help='save per epoch toggle',
                        dest='save_per_epoch', action='store_true')
    parser.set_defaults(save_per_epoch=False)

    parser.add_argument('-nld', '--no_learning_decay', help='disable decay on plaetu', action='store_false',
                        dest='lr_decay')
    parser.set_defaults(lr_decay=True)

    parser.add_argument('-es', '--early_stop', help='early stop', action='store_true', dest='early_stop')
    parser.set_defaults(early_stop=False)

    parser.add_argument('-tfdbg', '--tf_debug', help='TFDBG mode', action='store_true', dest='tf_dbg')
    parser.set_defaults(tf_dbg=False)

    parser.add_argument('-lw', '--load_weights', help='set for training from scratch', action='store_true',
                        dest='load_weights')
    parser.set_defaults(load_weights=False)
    return parser


def get_data_generator_flags(flag, target_size, data_config, mode='train'):
    image_gen_kwargs = data_config.train_image_gen_configs \
        if mode == 'train' else data_config.valid_image_gen_configs

    if str(flag).lower().find('vgg') >= 0:
        # if model_config.model_id == 'first_order':
        # print('First order set to resnet image gen')
        # data.image_data_generator = get_resnet_image_gen(model_config.target_size,
        #                                                  running_config.rescale_small,
        #                                                  running_config.random_crop,
        #                                                  running_config.horizontal_flip)
        # else:
        image_data_generator = get_vgg_image_gen(target_size, **image_gen_kwargs)
    elif str(flag).lower().find('densenet') >= 0:
        image_data_generator = get_densenet_image_gen(target_size,**image_gen_kwargs)
    else:
        image_data_generator = get_resnet_image_gen(target_size, **image_gen_kwargs)
    return image_data_generator


def get_data_generator(model_config, data_config, mode='train'):
    """ train or validation """
    return get_data_generator_flags(str(model_config.class_id).lower(),
                                    model_config.target_size,
                                    data_config,
                                    mode)


def finetune_with_model_data(data, model_config, dirhelper, nb_epoch_finetune, running_config):
    # TODO simplify this to fit the fintune_with_model_data_without_config
    """
    Generic training pipeline provided with data, model_config and nb_epoch_finetune

    Parameters
    ----------
    data
    model_config
    nb_epoch_finetune
    running_config

    Returns
    -------

    """
    # TODO Add the distributed to running_config if it works
    try:
        import horovod.tensorflow as hvd
        enable_distributed = True
    except ImportError as e:
        print("No hvd module, set enable_distributed to False")
        enable_distributed = False

    model_config.nb_class = data.nb_class
    # Get data generator
    if running_config.train_image_gen_configs:
        data.train_image_gen_configs = running_config.train_image_gen_configs
    if running_config.valid_image_gen_configs:
        data.valid_image_gen_configs = running_config.valid_image_gen_configs

    train_image_gen = get_data_generator(model_config, data, mode='train')
    valid_image_gen = get_data_generator(model_config, data, mode='valid')
    dirhelper.build(running_config.title + model_config.name)

    if nb_epoch_finetune > 0:
        # model_config2 = copy.copy(model_config)
        model_config.freeze_conv = True
        model = get_model(model_config)

        trainer = ClassificationTrainer(model, data, dirhelper,
                                        model_config=model_config, running_config=running_config,
                                        save_log=True,
                                        logfile=dirhelper.get_log_path(),
                                        train_image_gen=train_image_gen,
                                        valid_image_gen=valid_image_gen,
                                        enable_distributed=enable_distributed)
        trainer.build()
        # trainer.model.summary()
        trainer.fit(nb_epoch=nb_epoch_finetune, verbose=running_config.verbose)
        trainer.plot_result()
        # trainer.plot_model()
        model_config.freeze_conv = False
        running_config.load_weights = True
        running_config.init_weights_location = dirhelper.get_weight_path()

    elif nb_epoch_finetune == 0:
        model_config.freeze_conv = False
        model = get_model(model_config)
        trainer = ClassificationTrainer(model, data, dirhelper,
                                        model_config=model_config, running_config=running_config,
                                        save_log=True,
                                        logfile=dirhelper.get_log_path(),
                                        train_image_gen=train_image_gen,
                                        valid_image_gen=valid_image_gen,
                                        enable_distributed=enable_distributed)

    else:
        raise ValueError("nb_finetune_epoch must be non-negative {}".format(nb_epoch_finetune))

    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, BatchNormalization):
            layer.trainable = True

    # model = get_model(model_config)
    # trainer.model.summary()
    if nb_epoch_finetune > 0:
        # Evaluate before proceed.
        test_data = data.get_test()
        history = trainer.model.evaluate_generator(test_data, steps=test_data.n / running_config.batch_size)
        print("evaluation before re-training loss {} acc {}".format(history[0], history[1]))

    # Set the learning rate to 1/10 of original one during the finetune process.
    running_config.optimizer = SGD(lr=running_config.lr / 10, momentum=0.9, decay=0.)
    running_config.lr /= 10
    trainer.build()
    trainer.fit(verbose=running_config.verbose)
    trainer.plot_result()


def finetune_with_model_and_data_without_config(data, model, target_size, dirhelper, nb_epoch_finetune, running_config):
    """
       Generic training pipeline provided with data, model_config and nb_epoch_finetune

       Parameters
       ----------
       data
       model
       target_size (to get the generator)
       nb_epoch_finetune
       running_config

       Returns
       -------

       """

    nb_class = data.nb_class
    # Get data generator
    data.image_data_generator = get_data_generator_flags(model.name, target_size, running_config)
    dirhelper.build(running_config.title + model.name)

    if nb_epoch_finetune > 0:

        trainer = ClassificationTrainer(model, data, dirhelper,
                                        model_config=None, running_config=running_config,
                                        save_log=True,
                                        logfile=dirhelper.get_log_path())

        trainer.model.summary()
        trainer.fit(nb_epoch=nb_epoch_finetune, verbose=running_config.verbose)
        trainer.plot_result()
        running_config.load_weights = True
        running_config.init_weights_location = dirhelper.get_weight_path()

    elif nb_epoch_finetune == 0:
        trainer = ClassificationTrainer(model, data, dirhelper,
                                        model_config=None, running_config=running_config,
                                        save_log=True,
                                        logfile=dirhelper.get_log_path())

    else:
        raise ValueError("nb_finetune_epoch must be non-negative {}".format(nb_epoch_finetune))

    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, BatchNormalization):
            layer.trainable = True

    trainer.model.summary()
    if nb_epoch_finetune > 0:
        # Evaluate before proceed.
        test_data = data.get_test()
        history = trainer.model.evaluate_generator(test_data, steps=test_data.n / running_config.batch_size)
        print("evaluation before re-training loss {} acc {}".format(history[0], history[1]))

    # Set the learning rate to 1/10 of original one during the finetune process.
    running_config.optimizer = SGD(lr=running_config.lr / 10, momentum=0.9, decay=0.)
    trainer.build()
    trainer.fit(verbose=running_config.verbose)
    trainer.plot_result()
