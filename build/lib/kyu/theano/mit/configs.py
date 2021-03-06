from kyu.theano.general.config import DCovConfig


def get_new_experiment(exp):
    batch_size = 128
    cov_mode = 'pmean'
    mode_list = [1]
    cov_branch = 'o2t_no_wv'
    early_stop = False
    cov_regularizer = None
    last_config_feature_maps = [512]
    robust = True
    regroup = False
    cov_alpha = 0.3
    # cov_beta = 0.1
    cov_beta = 0.1
    pooling = 'max'
    if exp == 1:
        nb_branch = 2
        params = [[257, 128, 64], ]
        cov_outputs = [params[0][2]]
    elif exp == 2:
        nb_branch = 2
        params = [[256, 512, 512], ]
        cov_outputs = [params[0][2]]
    elif exp == 3:
        nb_branch = 2
        params = [[256,128,64,32]]
        cov_outputs = [params[0][2]]
    elif exp == 4:
        nb_branch = 2
        params = [[512, 256, 128, 64,]]
        cov_outputs = [params[0][2]]
        last_config_feature_maps = [1024]
    elif exp == 5:
        nb_branch = 4
        params = [[256, 128, 64, 32]]
        cov_outputs = [params[0][2]]
    elif exp == 6:
        nb_branch = 2
        params = [[256, 128, 64, 32]]
        cov_outputs = [params[0][2]]
        pooling = 'avg'
    else:
        return

    title = 'mit_indoor_newexperiment_{}_{}_{}'.format(cov_mode, cov_branch, pooling)
    # weight_path = '/home/kyu/.keras/models/mit_indoor_baseline_resnet50.weights'
    weight_path = 'imagenet'
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        cov_beta=cov_beta,
                        weight_path=weight_path,
                        pooling=pooling)
    return config


def get_experiment_settings(exp=1):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    if exp == 1:
        params = [[], [1024], [512], [1024, 512], [512, 256], [2048, 1444], [2048, 1024, 512]]
        mode_list = [1]
        cov_outputs = [512, 256, 128, 64]
        cov_branch = 'o2transform'
        cov_mode = 'channel'
        early_stop = True
    elif exp == 2:
        params = [[1024], [512], [1024, 512], [512, 256], [2048, 1444], [2048, 1024, 512]]
        mode_list = [1]
        cov_outputs = [512, 256, 128, 64]
        cov_branch = 'o2transform'
        cov_mode = 'mean'
        early_stop = True
    elif exp == 3:
        """Test the Regularizer """
        params = [[]]
        mode_list = [1]
        cov_outputs = [512, 256, 128]
        cov_mode = 'channel'
        cov_branch = 'o2transform'
        early_stop = True
        cov_regularizer = 'Fob'
    elif exp == 4:
        """Test VGG16 with DCov-2 """
        params = [[256, 128]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
        # last_config_feature_maps = [256]
        batch_size = 32

    elif exp == 5:
        """ Test ResNet 50 """
        params = [[256, 128], [256, 128, 64], [256], [128], [256, 256, 128],]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
        batch_size = 32
    elif exp == 6:
        """ Test Multi branch ResNet 50 """
        # nb_branch = 4
        nb_branch = 8
        params = [[256, 128], [256, 128, 64],]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [128, 64]
        cov_mode = 'mean'
        cov_branch = 'o2transform'
        early_stop = True
        # cov_regularizer = 'Fob'
        last_config_feature_maps = []
        # last_config_feature_maps = [1024, 512]
        batch_size = 32
    else:
        return
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp)
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
        params = [[200, 100, 50]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [50]
        cov_mode = 'pmean'
        cov_branch = 'aaai'
        early_stop = False
        robust = False
        # cov_regularizer = 'Fob'
        # last_config_feature_maps = []
        last_config_feature_maps = []
        # last_config_feature_maps = [1024, 512, 256]
        batch_size = 32
        vectorization = 'dense'

    elif exp == 2:
        """ aaai paper test """
        nb_branch = 2
        # params = [[513, 513, 513], [256, 256, 256]]
        # params = [[513, 513, 513, 513, 513, 513], [256, 256, 256, 256, 256]]
        params = [[200, 100, 50]]
        # params = [[1024, 512], [1024, 512, 256], [512, 256]]
        mode_list = [1]
        cov_outputs = [50]
        cov_mode = 'mean'
        cov_branch = 'aaai'
        early_stop = False
        robust = False
        # cov_regularizer = 'Fob'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024, 512]
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
        cov_mode = 'pmean'
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
    title = 'minc_von_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
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
        params = [[513, 257, 128, 64], ]
        # params = [[512, 256, 128, 64], ]
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
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
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
    title = 'minc_von_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup)
    return config




def get_VGG_dimension_reduction(exp=1):
    """ Test VGG dimension reduction """
    cov_regularizer = None
    if exp == 1:
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_branch = 'o2transform'
        cov_regularizer = None
        last_config_feature_maps = [1024]
        concat = 'concat'
    elif exp == 2:
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_branch = 'o2transform'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        concat = 'concat'
    elif exp == 3:
        nb_branch = 4
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_branch = 'o2transform'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        concat = 'concat'
    elif exp == 4:
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_branch = 'o2transform'
        cov_regularizer = None
        last_config_feature_maps = [1024]
        concat = 'sum'
    elif exp == 5:
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_branch = 'o2t_no_wv'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        concat = 'sum'
    elif exp == 6:
        nb_branch = 2
        params = [[257, 128, 64], ]
        mode_list = [1]
        cov_outputs = [64]
        cov_branch = 'o2t_no_wv'
        # last_config_feature_maps = []
        last_config_feature_maps = [1024]
        concat = 'avg'
    else:
        return
    cov_mode = 'pmean'
    early_stop = True
    batch_size = 32
    robust = True
    regroup = False
    cov_alpha = 0.75
    if robust:
        rb = 'robost'
    else:
        rb = ''
    title = 'minc2500_DR_{}_{}_LC{}'.format(cov_mode, rb, last_config_feature_maps)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        concat=concat)
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
    title = 'minc_von_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        concat=concat)
    return config


def get_matrix_bp(exp=1):
    cov_regularizer = None
    nb_branch = 1
    last_config_feature_maps = []
    batch_size = 4
    cov_alpha = 0.01
    if exp == 1:
        """ Test get matrix bp learning """
        nb_branch = 1
        params = [[]]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'channel'
        cov_branch = 'matbp'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        # last_config_feature_maps = []
        last_config_feature_maps = [512]
        batch_size = 32
        robust = True
        regroup = False
        cov_alpha = 0.75
        concat = 'concat'
    elif exp == 2:
        """ Test get matrix back prop with multi branch """
        nb_branch = 2
        params = [[]]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'matbp'
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
        nb_branch = 4
        params = [[257, 128, 64], ]
        mode_list = [3]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'matbp'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        last_config_feature_maps = []
        # last_config_feature_maps = [1024]
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
    title = 'mit_matbp_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        concat=concat)
    return config


def get_matrix_bp_vgg(exp=1):
    if exp == 1:
        """ Test get matrix bp learning """
        nb_branch = 1
        params = [[]]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'matbp'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        last_config_feature_maps = [512]
        # last_config_feature_maps = [1024, 512]
        batch_size = 32
        robust = False
        regroup = False
        cov_alpha = 0.75
        concat = 'concat'
    elif exp == 2:
        """ Test get matrix back prop with multi branch """
        nb_branch = 2
        params = [[]]
        mode_list = [1]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'matbp'
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
        nb_branch = 4
        params = [[257, 128, 64], ]
        mode_list = [3]
        cov_outputs = [64]
        cov_mode = 'mean'
        cov_branch = 'matbp'
        early_stop = False
        # cov_regularizer = 'vN'
        cov_regularizer = None
        last_config_feature_maps = []
        # last_config_feature_maps = [1024]
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
    title = 'minc_baseline_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
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
    title = 'minc_{}_{}_{}_{}'.format(cov_mode, cov_branch, rb, cov_regularizer)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, title=title, robust=robust, cov_alpha=cov_alpha)
    return config


def get_VGG_testing_ideas(exp):
    """ Test VGG dimension reduction """
    cov_regularizer = None
    if exp == 1:
        """ Experiment 1, cross validate number of branches. """
        nb_branch = 2
        # params = [[128, 64, 32], ]
        params = [[257, 128, 64], ]
        # params = [[64, 32, 16], ]
        mode_list = [1]
        # cov_outputs = [16]
        cov_outputs = [params[0][2]]
        cov_branch = 'o2transform'
        cov_regularizer = None
        last_config_feature_maps = [512]
        # last_config_feature_maps = [1024]
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
    title = 'minc2500_VGGTEST_{}_{}_LC{}_exp_{}_{}'.format(cov_branch, rb, last_config_feature_maps, exp, concat)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        concat=concat,
                        )
    return config


def get_ResNet_testing_ideas(exp):
    """ Test VGG dimension reduction """
    cov_regularizer = None
    if exp == 1:
        """ Experiment 1, cross validate number of branches. """
        nb_branch = 2
        # params = [[128, 64, 32], ]
        params = [[257, 128, 64], ]
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
        """ Experiment 2, no robust """
        nb_branch = 2
        # params = [[128, 64, 32], ]
        params = [[257, 128, 64], ]
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
        robust = False
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
    title = 'minc2500_RsNTEST_{}_{}_LC{}_exp_{}_{}'.format(cov_branch, rb, last_config_feature_maps, exp, concat)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        concat=concat,
                        )
    return config


def get_cov_alpha_cv(exp):
    if exp == 1:
        nb_branch = 1
        params = [[257, 128, 64], ]
        cov_outputs = [params[0][2]]
    else:
        return
    batch_size = 128
    cov_mode = 'pmean'
    mode_list = [1]
    cov_branch = 'o2transform'
    early_stop = True
    cov_regularizer = None
    last_config_feature_maps = [512]
    robust = True
    regroup = False
    cov_alpha = 0.4
    # cov_beta = 0.1
    cov_beta = 0.5

    title = 'minc2500_cv_covAlpha_{}_{}'.format(cov_alpha, cov_branch)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        cov_beta=cov_beta)
    return config


def get_cov_beta_cv(exp):
    if exp == 1:
        nb_branch = 1
        params = [[257, 128, 64], ]
        cov_outputs = [params[0][2]]
    else:
        return
    batch_size = 128
    cov_mode = 'pmean'
    mode_list = [1]
    cov_branch = 'o2transform'
    early_stop = True
    cov_regularizer = None
    last_config_feature_maps = [512]
    robust = True
    regroup = False
    cov_alpha = 0.75
    # cov_beta = 0.1
    cov_beta = 0.5

    title = 'minc2500_cv_covBeta_{}_{}'.format(cov_beta, cov_branch)
    config = DCovConfig(params, mode_list, cov_outputs, cov_branch, cov_mode, early_stop, cov_regularizer,
                        nb_branch=nb_branch, last_conv_feature_maps=last_config_feature_maps, batch_size=batch_size,
                        exp=exp, epsilon=1e-5, title=title, robust=robust, cov_alpha=cov_alpha, regroup=regroup,
                        cov_beta=cov_beta)
    return config
