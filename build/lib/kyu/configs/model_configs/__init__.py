from iccv_pow_transform import MPNConfig, BilinearSOConfig
from bilinear import BilinearConfig
from second_order import NoWVBranchConfig, O2TBranchConfig, NormWVBranchConfig, NewNormWVBranchConfig
from first_order import VggFirstOrderConfig, DenseNetFirstOrderConfig, ResNetFirstOrderConfig

MODEL_CONFIG_CLASS = {'mpn': MPNConfig,
                      'bilinear': BilinearConfig,
                      'sobilinear': BilinearSOConfig,
                      'nowv': NoWVBranchConfig,
                      'o2t': O2TBranchConfig,
                      'vggfo' : VggFirstOrderConfig,
                      'resnetfo': ResNetFirstOrderConfig,
                      'densenetfo': DenseNetFirstOrderConfig,
                      'normwv': NormWVBranchConfig,
                      'newnormwv': NewNormWVBranchConfig
                      }


def get_model_config_by_name(identifier):
    identifier = str(identifier).lower()
    if identifier in MODEL_CONFIG_CLASS.keys():
        return MODEL_CONFIG_CLASS[identifier]
    else:
        raise ValueError("model_config: cannot recognize the model config identifier {}".format(identifier))

