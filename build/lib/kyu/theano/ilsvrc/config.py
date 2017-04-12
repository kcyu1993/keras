from kyu.theano.general.config import DCovConfig


def get_VGG_testing_ideas(exp):
    """ Test VGG dimension reduction """
    cov_regularizer = None
    if exp == 1:
        """ Experiment 1, cross validate number of branches. """
        nb_branch = 4
        # params = [[128, 64, 32], ]
        # params = [[257, 128, 64], ]
        params = [[257, 128, 128], ]
        # params = [[64, 32, 16], ]
        mode_list = [1]
        # cov_outputs = [16]
        cov_outputs = [params[0][2]]
        cov_branch = 'o2t_no_wv'
        cov_regularizer = None
        # last_config_feature_maps = [512]
        last_config_feature_maps = [1024]
        concat = 'concat'
        last_avg = False
        robust = True
    elif exp == 2:
        """ Cross Validate on the summing methods """
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_branch = 'o2transform'
        cov_regularizer = None
        last_config_feature_maps = [512]
        # last_config_feature_maps = [1024]
        concat = 'ave'
        last_avg = False
        robust = False
    elif exp == 3:
        """ Test with robust single branch"""
        nb_branch = 1
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_branch = 'o2transform'
        cov_regularizer = None
        # last_config_feature_maps = []
        last_config_feature_maps = [512]
        concat = 'concat'
        last_avg = False
        robust = True
    elif exp == 4:
        """ Cross Validate on the summing methods for Riemannian"""
        nb_branch = 2
        # params = [[128, 64, 32],]
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_branch = 'o2t_no_wv'
        cov_regularizer = None
        last_config_feature_maps = [512]
        # last_config_feature_maps = [1024]
        concat = 'concat'
        last_avg = False
        robust = True
    else:
        return
    cov_mode = 'pmean'
    early_stop = True
    batch_size = 32

    regroup = False
    cov_alpha = 0.75
    if robust:
        rb = 'robost'
    else:
        rb = ''
    cov_beta = 0.3
    title = 'ImageNet_VGG16_{}_{}_LC{}_exp_{}_{}'.format(cov_branch, rb, last_config_feature_maps, exp, concat)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        concat=concat, cov_beta=cov_beta,
                        )
    return config
