# Test the simple config

import numpy as np
from kyu.models import get_model
from kyu.configs.experiment_configs.simple_second_order_config import *
from kyu.configs.experiment_configs.running_configs import *


def test_o2t_wv_new_norm_with_setting():
    np.cumsum()
    model_config = get_new_wv_norm_general(14)
    model_config.class_id = 'vgg16'

    # running_config = get_running_config_no_debug_withSGD()

    # assume the input size
    model = get_model(model_config)
    model.compile('SGD', 'categorical_crossentropy')
    model.summary()

if __name__ == '__main__':
    test_o2t_wv_new_norm_with_setting()