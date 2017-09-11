from iccv_pow_transform import MPNConfig
from bilinear import BilinearConfig
from second_order import NoWVBranchConfig, O2TBranchConfig
from first_order import VggFirstOrderConfig, DenseNetFirstOrderConfig, ResNetFirstOrderConfig

MODEL_CONFIG_CLASS = {'mpn': MPNConfig,
                      'bilinear': BilinearConfig,
                      'nowv': NoWVBranchConfig,
                      'o2t': O2TBranchConfig,
                      'vggfo' : VggFirstOrderConfig,
                      'resnetfo': ResNetFirstOrderConfig,
                      'densenetfo': DenseNetFirstOrderConfig
                      }


def get_model_config_by_name(identifier):
    identifier = str(identifier).lower()
    if identifier in MODEL_CONFIG_CLASS.keys():
        return MODEL_CONFIG_CLASS[identifier]
    else:
        raise ValueError("model_config: cannot recognize the model config identifier {}".format(identifier))

