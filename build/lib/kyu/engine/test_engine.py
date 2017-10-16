"""
Testing for the trainer built

Use DTD and SO-VGG16 for example

"""
from kyu.configs.engine_configs import RunningConfig
from kyu.configs.model_configs import MPNConfig
from kyu.configs.model_configs.bilinear import BilinearConfig
from kyu.datasets.dtd import DTD
from kyu.engine.trainer import ClassificationTrainer
from kyu.experiment.configuration import get_running_config
from kyu.experiment.data_train_utils import dtd_finetune_with_model
from kyu.models import get_model
from kyu.utils.io_utils import ProjectFile


def get_dtd_bilinear_model():
    data = DTD('/home/kyu/.keras/datasets/dtd', name='DTD_test')

    nb_class = data.nb_class
    input_shape = (224, 224, 3)

    # Initialize the model
    model_config = BilinearConfig(nb_class, input_shape)

    model = get_model(model_config)

    running_config = RunningConfig(
        'Bilinear-Test-with-log', nb_epoch=3, batch_size=32
    )

    # Initialize the file helper
    dirhelper = ProjectFile(root_path='/home/kyu/cvkyu/test_bilinear', dataset=data.name, model_category='VGG16')

    dirhelper.build(running_config.title)
    return model, data, dirhelper, model_config, running_config


def get_dtd_so_model():
    pass


def test_dtd_bilinear():
    """

    Returns
    -------

    """
    # initialize the data
    model, data, dirhelper, model_config, running_config = get_dtd_bilinear_model()

    # Initialize the trainer
    trainer = ClassificationTrainer(model, data, dirhelper, model_config, running_config)
    trainer.fit()
    trainer.plot_result()
    trainer.plot_model()


def test_model_plot_function():
    model, data, dirhelper, model_config, running_config = get_dtd_bilinear_model()
    trainer = ClassificationTrainer(model, data, dirhelper, model_config, running_config)
    trainer.plot_model()


def test_tfdbg_session():
    from tensorflow.python import debug as tfdbg

    model_config = MPNConfig(input_shape=(224, 224, 3),
                             nb_class=67,
                             parametric=[128],
                             activation='relu',
                             cov_mode='channel',
                             vectorization='wv',
                             epsilon=1e-5,
                             use_bias=True,
                             cov_alpha=0.1,
                             cov_beta=0.3,
                             normalization=None,
                             mode=1,
                             last_conv_feature_maps=[256],
                             cov_branch_output=128)

    running_config = get_running_config('DTD-test-with-tfdbg', model_config)

    running_config.tf_debug = True
    running_config.tf_debug_filters_func = [tfdbg.has_inf_or_nan]
    running_config.tf_debug_filters_name = ['has_inf_or_nan']

    dtd_finetune_with_model(model_config, 0, running_config)


if __name__ == '__main__':
    # test_sun397()
    test_dtd_bilinear()