from kyu.engine.trainer import ClassificationTrainer
from kyu.models import get_model
from kyu.utils.image import get_vgg_image_gen, get_resnet_image_gen
from kyu.utils.io_utils import ProjectFile


def get_dirhelper(dataset_name, model_category, **kwargs):
    return ProjectFile(root_path='/home/kyu/cvkyu/secondstat', dataset=dataset_name, model_category=model_category,
                       **kwargs)


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
        data.image_data_generator = get_vgg_image_gen(model_config.target_size,
                                                      running_config.rescale_small,
                                                      running_config.random_crop,
                                                      running_config.horizontal_flip)
    else:
        data.image_data_generator = get_resnet_image_gen(model_config.target_size,
                                                         running_config.rescale_small,
                                                         running_config.random_crop,
                                                         running_config.horizontal_flip)
    dirhelper.build(running_config.title)

    if nb_epoch_finetune > 0:
        # model_config2 = copy.copy(model_config)
        model_config.freeze_conv = True
        model = get_model(model_config)

        trainer = ClassificationTrainer(model, data, dirhelper,
                                        model_config=model_config, running_config=running_config,
                                        save_log=True,
                                        logfile=dirhelper.get_log_path())

        trainer.model.summary()
        trainer.fit(nb_epoch=nb_epoch_finetune, verbose=2)
        trainer.plot_result()
        # trainer.plot_model()
        model_config.freeze_conv = False
        running_config.load_weights = True
        running_config.init_weights_location = dirhelper.get_weight_path()

    model = get_model(model_config)
    if nb_epoch_finetune == 0:
        model.summary()

    trainer = ClassificationTrainer(model, data, dirhelper,
                                    model_config=model_config, running_config=running_config,
                                    save_log=True,
                                    logfile=dirhelper.get_log_path())

    trainer.build()

    trainer.fit(verbose=2)
    trainer.plot_result()

