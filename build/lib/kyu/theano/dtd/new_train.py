"""
New Train pipeline with trainer

"""
from keras.optimizers import SGD
from kyu.configs.engine_configs import RunningConfig
from kyu.configs.model_configs.first_order import VggFirstOrderConfig
from kyu.datasets.dtd import DTD
from kyu.theano.dtd.configs import get_o2t_testing
from kyu.theano.general.train import finetune_with_model_data
from kyu.utils.io_utils import ProjectFile


def get_test_optimizer():
    """ Get a SGD with gradient clipping """
    return SGD(lr=0.01, momentum=0.9, decay=1e-4, clipvalue=1e4)


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


def get_vgg_config():
    return VggFirstOrderConfig(
        nb_classes=67,
        input_shape=(224, 224, 3),
        include_top=True,
        freeze_conv=False
    )


def finetune_with_model(model_config, nb_epoch_finetune, running_config):
    data = DTD('/home/kyu/.keras/datasets/dtd', name='DTD')
    dirhelper = ProjectFile(root_path='/home/kyu/cvkyu/secondstat', dataset=data.name, model_category='VGG16')
    finetune_with_model_data(data, model_config, dirhelper, nb_epoch_finetune, running_config)


def so_vgg_experiment(exp=1):
    title = 'DTD-Testing-o2transform'
    model_config = get_o2t_testing(exp)
    running_config = get_running_config(title, model_config)
    running_config.rescale_small = 256
    finetune_with_model(model_config, nb_epoch_finetune=10, running_config=running_config)


def test_minc_experiment(exp=1):
    pass


def vgg_baseline(exp=1):
    baseline_model_config = get_vgg_config()
    running_config = get_running_config('DTD-VGG6-baseline-withtop', baseline_model_config)
    finetune_with_model(baseline_model_config, 4, running_config)


if __name__ == '__main__':
    so_vgg_experiment(2)
    # vgg_baseline()
    # mpn_cov_baseline(1)
