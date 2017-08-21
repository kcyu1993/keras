"""
Testing for the trainer built

Use DTD and SO-VGG16 for example

"""
from kyu.engine.configs.model import ModelConfig
from kyu.engine.configs.running import RunningConfig
from kyu.engine.configs.generic import KCConfig

from kyu.utils.io_utils import ProjectFile

from kyu.datasets.dtd import DTD
from kyu.models.bilinear import VGG16_bilinear
from kyu.engine.trainer import ClassificationTrainer

from kyu.models import get_model


class BilinearConfig(ModelConfig):

    def __init__(self,
                 nb_class,
                 input_shape,
                 ):
        super(BilinearConfig, self).__init__('vgg', 'bilinear')
        self.nb_class = nb_class
        self.input_shape = input_shape
        self.load_weights = 'imagenet'


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


if __name__ == '__main__':
    # test_sun397()
    test_dtd_bilinear()