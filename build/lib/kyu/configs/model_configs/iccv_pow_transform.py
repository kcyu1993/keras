"""
Implement the Config for MPN-Cov network

"""
from .second_order import DCovConfig


class MPNConfig(DCovConfig):

    def __init__(self,
                 parametric=[],
                 activation='relu',
                 cov_mode='channel',
                 vectorization='mat_flatten',
                 epsilon=1e-5,
                 use_bias=False,
                 cov_alpha=0.1,
                 cov_beta=0.3,
                 normalization=None,
                 **kwargs
                 ):
        cov_branch = 'pow_o2t'
        z = {"parametric": parametric, 'activation': activation, 'parametric': parametric, 'cov_mode': cov_mode,
             'vectorization': vectorization, 'epsilon': epsilon, 'use_bias': use_bias, 'cov_alpha': cov_alpha,
             'cov_beta': cov_beta, 'normalization': normalization}
        super(MPNConfig, self).__init__(cov_branch=cov_branch, cov_branch_kwargs=z, **kwargs)

