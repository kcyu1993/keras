from configobj import ConfigObj

from kyu.configs.engine_configs import ModelConfig


class DCovConfig(ModelConfig):

    def __init__(self,
                 input_shape,
                 cov_branch,
                 cov_branch_kwargs,
                 nb_class=0,
                 load_weights='imagenet',
                 # support just in case.
                 class_id=None,
                 model_id=None,
                 # configs for _compose_second_order_things
                 mode=0, cov_branch_output=None,
                 freeze_conv=False, name='default_so_model',
                 nb_branch=1,
                 concat='concat',
                 cov_output_vectorization='pv',
                 last_conv_feature_maps=[],
                 last_conv_kernel=[1, 1],
                 upsample_method='conv',
                 **kwargs
                 ):
        model_id = 'second_order'

        super(DCovConfig, self).__init__(class_id=class_id, model_id=model_id, **kwargs)
        self.__dict__.update(locals())

    @classmethod
    def recover_from_parameter(cls, cov_branch, cov_branch_kwargs, **kwargs):
        config = cls(**kwargs)
        config.cov_branch = cov_branch
        config.cov_branch_kwargs = cov_branch_kwargs
        return config

    @classmethod
    def recover_from_config(cls, infile):
        config = ConfigObj(infile, raise_errors=True)
        if 'self' in config.keys():
            del config['self']
        if 'kwargs' in config.keys():
            del config['kwargs']

        constructor_dict = config.dict()
        cov_branch = constructor_dict['cov_branch']
        del constructor_dict['cov_branch']
        cov_branch_kwargs = constructor_dict['cov_branch_kwargs']
        del constructor_dict['cov_branch_kwargs']

        this_config = cls(**constructor_dict)
        this_config.cov_branch = cov_branch
        this_config.cov_branch_kwargs = cov_branch_kwargs
        return this_config

    @classmethod
    def load_config_from_file(cls, infile):
        config = ConfigObj(infile, raise_errors=True)
        if 'self' in config.keys():
            del config['self']
        return cls(**config.dict())


class O2TBranchConfig(DCovConfig):
    def __init__(self,
                 parametric=[],
                 activation='relu',
                 cov_mode='channel',
                 vectorization='wv',
                 epsilon=1e-5,
                 use_bias=True,
                 robust=True,
                 cov_alpha=0.1,
                 cov_beta=0.3,
                 o2t_constraints=None,
                 o2t_regularizer=None,
                 o2t_activation='relu',
                 **kwargs
                 ):
        cov_branch = 'o2transform'
        z = {"parametric": parametric, 'activation': activation, 'parametric': parametric, 'cov_mode': cov_mode,
             'vectorization': vectorization, 'epsilon': epsilon, 'use_bias': use_bias, 'cov_alpha': cov_alpha,
             'cov_beta': cov_beta, 'robust': robust,
             'o2t_regularizer': o2t_regularizer,
             'o2t_activation': o2t_activation,
             'o2t_constraints': o2t_constraints}

        super(O2TBranchConfig, self).__init__(cov_branch=cov_branch, cov_branch_kwargs=z, **kwargs)



class NoWVBranchConfig(DCovConfig):
    def __init__(self,
                 parametric=[],
                 epsilon=1e-7,
                 activation='relu',
                 cov_mode='channel',
                 cov_alpha=0.3,
                 cov_beta=0.1,
                 robust=False,
                 normalization=False,
                 o2t_constraints=None,
                 o2t_regularizer=None,
                 o2t_activation='relu',
                 **kwargs
                 ):
        cov_branch = 'o2t_no_wv'
        z = {"parametric": parametric, "epsilon": epsilon, "activation": activation,
             'cov_mode': cov_mode,
             'epsilon': epsilon, 'cov_alpha': cov_alpha,
             'cov_beta': cov_beta, 'robust': robust, "normalization":normalization,
             'o2t_regularizer': o2t_regularizer,
             'o2t_activation': o2t_activation,
             'o2t_constraints': o2t_constraints
             }
        super(NoWVBranchConfig, self).__init__(cov_branch=cov_branch, cov_branch_kwargs=z, **kwargs)