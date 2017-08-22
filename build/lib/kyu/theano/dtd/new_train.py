"""
New Train pipeline with trainer

"""
import copy

from kyu.datasets.dtd import DTD
from kyu.datasets.imagenet import preprocess_image_for_imagenet, preprocess_image_for_imagenet_without_channel_reverse
from kyu.engine.configs import RunningConfig
from kyu.engine.trainer import ClassificationTrainer
from kyu.models import get_model
from kyu.theano.dtd.configs import get_o2t_testing
from kyu.utils.image import ImageDataGeneratorAdvanced
from kyu.utils.io_utils import ProjectFile


def get_vgg_image_gen(target_size, rescale_small, random_crop=True, horizontal_flip=True):

    return ImageDataGeneratorAdvanced(
        target_size, rescale_small, random_crop=random_crop,
        horizontal_flip=horizontal_flip,
        preprocessing_function=preprocess_image_for_imagenet
    )


def get_resnet_image_gen(target_size, rescale_small, random_crop=True, horizontal_flip=True):
    return ImageDataGeneratorAdvanced(
        target_size, rescale_small, random_crop=random_crop,
        horizontal_flip=horizontal_flip,
        preprocessing_function=preprocess_image_for_imagenet_without_channel_reverse
    )


def get_running_config(title='DTD_testing', model_config=None):
    return RunningConfig(
        _title=title,
        nb_epoch=200,
        batch_size=32,
        verbose=2,
        lr_decay=False,
        sequence=8,
        patience=8,
        early_stop=False,
        save_weights=True,
        load_weights=False,
        init_weights_location=None,
        save_per_epoch=True,
        tensorboard=None,
        optimizer='SGD',
        lr=0.01,
        model_config=model_config,
        # Image Generator Config
        rescale_small=296,
        random_crop=True,
        horizontal_flip=True,
    )


def finetune_with_model(model_config, nb_epoch_finetune, running_config):
    data = DTD('/home/kyu/.keras/datasets/dtd', name='DTD')
    # Update
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
    dirhelper = ProjectFile(root_path='/home/kyu/cvkyu/secondstat', dataset=data.name, model_category='VGG16')
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

    trainer = ClassificationTrainer(model, data, dirhelper,
                                    model_config=model_config, running_config=running_config,
                                    save_log=True,
                                    logfile=dirhelper.get_log_path())

    trainer.build()

    trainer.fit(verbose=2)
    trainer.plot_result()


if __name__ == '__main__':
    title = 'DTD-Testing-o2transform'
    model_config = get_o2t_testing(2)
    running_config = get_running_config(title, model_config)

    finetune_with_model(model_config, nb_epoch_finetune=10, running_config=running_config)


