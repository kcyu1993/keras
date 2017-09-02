import argparse

from keras.layers import Conv2D, BatchNormalization
from keras.optimizers import SGD
from kyu.engine.trainer import ClassificationTrainer
from kyu.models import get_model
from kyu.utils.image import get_vgg_image_gen, get_resnet_image_gen
from kyu.utils.io_utils import ProjectFile


def get_dirhelper(dataset_name, model_category, **kwargs):
    return ProjectFile(root_path='/home/kyu/cvkyu/so_updated_record', dataset=dataset_name, model_category=model_category,
                       **kwargs)


def get_argparser(description='default'):
    """
    Define the general arg-parser
    Returns
    -------

    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset name: support dtd, minc2500')
    parser.add_argument('-me', '--model_exp', help='model experiment index', type=int, default=1)
    parser.add_argument('-m', '--model_class', help='model class should be in vgg, resnet', default='vgg', type=str)
    parser.add_argument('-ef', '--nb_epoch_finetune', help='number of epoch to finetune', default=0, type=int)
    parser.add_argument('-et', '--nb_epoch_train', help='number of epoch to retrain', default=200, type=int)
    parser.add_argument('-dbg', '--debug', type=bool, help='True for entering TFDbg mode', default=False)
    parser.add_argument('-tb', '--tensorboard', type=bool, help='Enable Tensorboard monitoring', default=True)
    parser.add_argument('-c', '--comments', help='comments if any', default='', type=str)
    parser.add_argument('--channel_reverse', help='enable channel transform from RGB to BGR', default=False, type=bool)
    return parser


def finetune_with_model_data(data, model_config, dirhelper, nb_epoch_finetune, running_config):
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

    model_config.nb_class = data.nb_class
    if model_config.class_id == 'vgg':
        # if model_config.model_id == 'first_order':
            # print('First order set to resnet image gen')
            # data.image_data_generator = get_resnet_image_gen(model_config.target_size,
            #                                                  running_config.rescale_small,
            #                                                  running_config.random_crop,
            #                                                  running_config.horizontal_flip)
        # else:
        data.image_data_generator = get_vgg_image_gen(model_config.target_size,
                                                      running_config.rescale_small,
                                                      running_config.random_crop,
                                                      running_config.horizontal_flip)
    else:
        data.image_data_generator = get_resnet_image_gen(model_config.target_size,
                                                         running_config.rescale_small,
                                                         running_config.random_crop,
                                                         running_config.horizontal_flip)
    dirhelper.build(running_config.title + model_config.name)

    if nb_epoch_finetune > 0:
        # model_config2 = copy.copy(model_config)
        model_config.freeze_conv = True
        model = get_model(model_config)

        trainer = ClassificationTrainer(model, data, dirhelper,
                                        model_config=model_config, running_config=running_config,
                                        save_log=True,
                                        logfile=dirhelper.get_log_path())

        trainer.model.summary()
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
                                        logfile=dirhelper.get_log_path())

    else:
        raise ValueError("nb_finetune_epoch must be non-negative {}".format(nb_epoch_finetune))

    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, BatchNormalization):
            layer.trainable = True

    # model = get_model(model_config)
    trainer.model.summary()
    if nb_epoch_finetune > 0:
        # Evaluate before proceed.
        test_data = data.get_test()
        history = trainer.model.evaluate_generator(test_data, steps=test_data.n / running_config.batch_size)
        print("evaluation before re-training loss {} acc {}".format(history[0], history[1]))

    # Set the learning rate to 1/10 of original one during the finetune process.
    running_config.optimizer = SGD(lr=running_config.lr / 10, momentum=0.9, decay=1e-5)
    trainer.build()
    trainer.fit(verbose=running_config.verbose)
    trainer.plot_result()

