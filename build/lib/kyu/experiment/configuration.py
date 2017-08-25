from kyu.configs.engine_configs import RunningConfig
from kyu.configs.experiment_configs.running_configs import get_running_config_no_debug_withSGD


def get_running_config(title='general-testing', model_config=None):
    return get_running_config_no_debug_withSGD(title, model_config)

