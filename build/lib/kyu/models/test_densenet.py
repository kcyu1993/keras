"""
Test DenseNet Experiments
"""
from kyu.configs.model_configs import DenseNetFirstOrderConfig
from kyu.models import get_model


def test_densenet121():
    config = DenseNetFirstOrderConfig(397, freeze_conv=True)
    densenet = get_model(config)
    densenet.compile('SGD', 'categorical_crossentropy', ['acc',])
    densenet.summary()


if __name__ == '__main__':
    test_densenet121()




