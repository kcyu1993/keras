"""
Stores CIFAR config settings for experiments

"""

from kyu.theano.general.config import DCovConfig

def get_experiment_settings(exp=1):
    nb_branch = 1
    batch_size = 128
    if exp == 1:
        params = [[64, 32, 16], [25, 25, 25],]
        cov_outputs = [16]
        cov_mode = 'pmean'
    elif exp == 2:
        """ Test fitnet with simple settings """
        nb_branch = 2
        params = [[100, 50, 25], ]
        cov_outputs = [64]
        cov_mode = 'pmean'
    elif exp == 3:
        # params = [[64, 64, 64], [100, 100, 100], [50, 50, 50],]
        # params = [[64], [64, 128], [64,128,256],]
        params = [[64], [64, 32], [64,32,16],]
        cov_outputs = []
        cov_mode = 'pmean'
        title = 'cifar10_CrossV-WV_{}'.format(cov_mode, )
    elif exp == 4:
        params = [[30], [50], [70]]
        # params = [[], [50,], [70], [64,64,64]]
        # cov_outputs = range(10, 201, 20)
        cov_outputs = range(20, 201, 20)
        cov_mode = 'pmean'
        title = 'cifar10_CrossV-WV_{}'.format(cov_mode,)
    elif exp == 5:
        # params = [[50, 40, 30], [50, 60, 70], [50,30,10], [50, 70, 90]]
        # params = [[50, 100, 200, 400,],]
        # params = [[50,50,50,50],]
        # params = [[50,50,50,50, 50],]
        params = [[50, 100, 200, 400,],]
        # params = [[50, 100, 200, 400, 800],]
        # params = [[800,400,200,100,50]]
        # params = [[50, 50, 50]]
        # params = [[50, 50,], [50], [50,50,50]]
        # params = [[50, 50,], [50], [50,50,50]]
        # params = [[50, 100, 200], [50, 100], [100,50], [200,100,50]]
        cov_outputs = [params[0][-1]]
        cov_mode = 'pmean'
        title = 'cifar10-CrossV-O2T_{}_'.format(cov_mode)
    elif exp == 6:
        params = [[50, 25], [50, 75], ]
        cov_outputs = [25, 50, 75]
        cov_mode = 'pmean'
        title = 'cifar10-CrossV-O2T_{}_'.format(cov_mode)
    elif exp == 7:
        # params = [[64,64,64]]
        # params = [[190], [170],[110], [30 ]]
        # params = [[90]]
        params = [[70]]
        # params = [[50]]
        # params = [[30]]
        # cov_outputs = [200, 160, 80, 40] # 90
        # cov_outputs = [190, 170, ] # 70
        # cov_outputs = [20] # 50
        cov_outputs = [40,70,10] # 30
        # cov_outputs = [50]
        cov_mode = 'pmean'
        title = 'cifar10-CrossV-O2T_{}_'.format(cov_mode)
    elif exp == 8:
        # params = [[]]
        # params = [[150]]
        params = [[100]]
        # params = [[90]]
        # params = [[70]]
        # params = [[50]]
        # params = [[30]]
        # cov_outputs = range(20, 201, 20)
        cov_outputs = range(220, 401, 20)
        cov_mode = 'pmean'
        title = 'cifar10_CrossV-WV_{}'.format(cov_mode,)
    elif exp == 9:
        params = [[150]]
        # params = [[100]]
        # params = [[90]]
        # params = [[70]]
        # params = [[50]]
        # params = [[30]]
        cov_outputs = range(10, 201, 20)
        cov_mode = 'pmean'
        title = 'cifar10_CrossV-WV_{}'.format(cov_mode,)
    else:
        return

    mode_list = [1]
    cov_branch = 'o2transform'
    early_stop = True
    cov_regularizer = None
    last_config_feature_maps = []
    robust = False
    regroup = False
    cov_alpha = 1
    cov_beta = 0.1
    if robust:
        rb = 'robost'
    else:
        rb = ''

    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        cov_beta=cov_beta)
    return config


def get_cov_beta_cv(exp):
    nb_branch = 1
    if exp == 1:
        params = [[50,100,200]]
        # params = [[100]]
        # params = [[90]]
        # params = [[70]]
        # params = [[50]]
        # params = [[30]]
        cov_outputs = [200]
        cov_mode = 'pmean'
    else:
        return
    batch_size = 128
    mode_list = [1]
    cov_branch = 'o2transform'
    early_stop = True
    cov_regularizer = None
    last_config_feature_maps = []
    robust = False
    regroup = False
    cov_alpha = 1
    # cov_beta = 0.1
    cov_beta = 0.7

    title = 'cifar10_cv_covBeta_{}_{}'.format(cov_beta, cov_branch)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        cov_beta=cov_beta)
    return config


def get_aaai_experiment(exp):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    if exp == 1:
        """ aaai paper test """
        nb_branch = 1
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [[70, 50, 30]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [50]
        cov_mode = 'mean'
        cov_branch = 'aaai'
        early_stop = False
        robust = False
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        # last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
        vectorization = 'dense'

    elif exp == 2:
        """ aaai paper test """
        nb_branch = 2
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [[70, 50, 30]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [50]
        cov_mode = 'mean'
        cov_branch = 'aaai'
        early_stop = False
        robust = False
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        # last_config_feature_maps = [1024, 512]
        batch_size = 32
        vectorization = 'flatten'
    else:
        return
    title = 'aaai_baseline'
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, vectorization=vectorization, epsilon=1e-5, title=title, robust=robust)
    return config


def get_log_experiment(exp):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    if exp == 1:
        """ log test """
        nb_branch = 1
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [[513, 257, 129], [513, 513, 513]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'log'
        early_stop = False
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
        batch_size = 32
        vectorization = 'wv'
    elif exp == 2:
        """ aaai paper test """
        nb_branch = 1
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [[256, 128, 64], [513, 257, 129, 64]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'log'
        early_stop = False
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
        batch_size = 32
        vectorization = 'dense'
    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, vectorization=vectorization, epsilon=1e-5)
    return config


def get_von_settings(exp=1):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    cov_alpha = 0.01
    if exp == 1:
        """ Test Multi branch ResNet 50 """
        nb_branch = 1
        params = [[256, 128, 64],]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        # cov_regularizer = None
        # cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
        robust = True
        cov_alpha = 0.75
    elif exp == 2:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 4
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = False

    elif exp == 3:
        """ Test Multi branch ResNet 50 """
        nb_branch = 1
        params = [[256, 128, 64], ]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        cov_regularizer = None
        # cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
        robust = True
        cov_alpha = 0.75
    elif exp == 4:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        cov_regularizer = None
        # cov_regularizer = 'vN'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        cov_alpha = 0.75
    else:
        return

    if robust:
        rb = 'robost'
    else:
        rb = ''
    title = 'cifar10_von_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha)
    return config



def get_von_with_regroup(exp=1):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    cov_alpha = 0.01
    if exp == 1:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 4
        params = [[100, 50, 25], ]
        # params = [[512, 256, 128, 64], ]
        mode_list = [1]
        cov_outputs = [25]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        last_config_feature_maps = []
        # last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        regroup = False
        cov_alpha = 0.75
    elif exp == 2:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        last_config_feature_maps = []
        # last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        regroup = False
        cov_alpha = 0.75
    else:
        return

    if robust:
        rb = 'robost'
    else:
        rb = ''
    title = 'cifar10_von_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup)
    return config


def get_von_with_multibranch(exp=1):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    cov_alpha = 0.01
    if exp == 1:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 1
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2t_no_wv'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        regroup = False
        cov_alpha = 0.75
        concat = 'sum'
    elif exp == 2:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [2]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        regroup = False
        cov_alpha = 0.75
        concat = 'concat'
    elif exp == 3:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [3]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'o2t_no_wv'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        batch_size = 32
        robust = True
        regroup = False
        cov_alpha = 0.75
        concat = 'concat'
    else:
        return

    if robust:
        rb = 'robost'
    else:
        rb = ''
    title = 'cifar10_von_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        concat=concat)
    return config


def get_matrix_bp(exp=1):
    if exp == 1:
        """ Test get matrix bp learning """
        nb_branch = 1
        mode_list = [1]
    elif exp == 2:
        """ Test get matrix back prop with multi branch """
        nb_branch = 1
        mode_list = [1]
    elif exp == 3:
        """ Test Multi_branch Resnet 50 with residual learning """
        nb_branch = 4
        mode_list = [3]
    else:
        return
    params = [[]]
    cov_outputs = [64]
    cov_mode = 'channel'
    cov_branch = 'matbp'
    early_stop = True
    cov_regularizer = None
    last_config_feature_maps = []
    batch_size = 128
    robust = False
    regroup = False
    cov_alpha = 0.75
    concat = 'concat'
    if robust:
        rb = 'robost'
    else:
        rb = ''
    title = 'cifar10_matbp_basline_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        concat=concat, normalization=None)
    return config


def get_residual_cov_experiment(exp):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    if exp == 1:
        """ Test Multi branch ResNet 50 """
        nb_branch = 1
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [ [513, 513, 513, 513, 513, 513], [513, 513,], [513, 513,513, 513],[513, 513, 513, 513, 513,], ]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'residual'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
        batch_size = 32
    elif exp == 2:
        """ Test Multi branch ResNet 50 """
        nb_branch = 4
        params = [[513, 513, 513, 513, 513, 513], [513, 513, 513], [256, 256, 256]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'residual'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        # last_config_feature_maps = [1024]
        robust = True
        cov_alpha = 0.75
        batch_size = 32
    else:
        return
    if robust:
        rb = 'robost'
    else:
        rb = ''
    title = 'cifar10_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, title=title, robust=robust, cov_alpha=cov_alpha)
    return config


def get_resnet_batch_norm(exp):
    nb_branch = 1
    if exp == 1:
        params = [[]]
        # params = [[100]]
        # params = [[90]]
        # params = [[70]]
        # params = [[50]]
        # params = [[30]]
        cov_outputs = [200]
        cov_mode = 'pmean'
    else:
        return
    batch_size = 128
    mode_list = [1]
    cov_branch = 'o2transform'
    early_stop = True
    cov_regularizer = None
    last_config_feature_maps = []
    robust = False
    regroup = False
    cov_alpha = 1
    # cov_beta = 0.1
    cov_beta = 0.7

    title = 'cifar10_cv_covBeta_{}_{}'.format(cov_beta, cov_branch)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        cov_beta=cov_beta)
    return config
