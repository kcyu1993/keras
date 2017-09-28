"""
New Train pipeline with trainer

"""

from keras.optimizers import SGD
from kyu.configs.engine_configs import RunningConfig
from kyu.configs.model_configs.first_order import VggFirstOrderConfig
from kyu.datasets.dtd import DTD
from kyu.legacy.theano.general.train import finetune_with_model_data
from kyu.utils.io_utils import ProjectFile


def get_test_optimizer():
    """ Get a SGD with gradient clipping """
    return SGD(lr=0.01, momentum=0.9, decay=1e-4, clipvalue=1e4)


from kyu.configs.model_configs.second_order import O2TBranchConfig, NoWVBranchConfig


def get_o2t_testing(exp):
    """ Same as before, updated with the new framework """
    if exp == 1:
        return O2TBranchConfig(
            parametric=[128],
            activation='relu',
            cov_mode='pmean',
            cov_alpha=0.3,
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            input_shape=(224,224,3),
            nb_class=67,
            cov_branch='o2transform',
            cov_branch_output=128,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='TestingO2T-bias',
            nb_branch=1,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
    elif exp == 2:
        return O2TBranchConfig(
            parametric=[64, 32,],
            activation='relu',
            cov_mode='channel',
            cov_alpha=0.3,
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch='o2transform',
            cov_branch_output=32,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='Original-O2T-testing',
            nb_branch=2,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
    elif exp == 3:
        return O2TBranchConfig(
            parametric=[64, 32, ],
            activation='relu',
            cov_mode='channel',
            cov_alpha=0.3,
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch='o2transform',
            cov_branch_output=32,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='Original-O2T-testing',
            nb_branch=2,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )

    else:

        raise ValueError("N")


def get_running_config(title='DTD_testing', model_config=None):
    return RunningConfig(
        _title=title,
        nb_epoch=200,
        batch_size=32,
        verbose=2,
        lr_decay=False,
        sequence=8,
        patience=8,
        early_stop=False,
        save_weights=True,
        load_weights=False,
        init_weights_location=None,
        save_per_epoch=True,
        tensorboard=None,
        optimizer='SGD',
        lr=0.01,
        model_config=model_config,
        # Image Generator Config
        rescale_small=296,
        random_crop=True,
        horizontal_flip=True,
    )


def get_vgg_config():
    return VggFirstOrderConfig(
        nb_classes=67,
        input_shape=(224, 224, 3),
        include_top=True,
        freeze_conv=False
    )


def dtd_finetune_with_model(model_config, nb_epoch_finetune, running_config):
    data = DTD('/home/kyu/.keras/datasets/dtd', name='DTD')
    dirhelper = ProjectFile(root_path='/home/kyu/cvkyu/secondstat', dataset=data.name, model_category='VGG16')
    finetune_with_model_data(data, model_config, dirhelper, nb_epoch_finetune, running_config)


def so_vgg_experiment(exp=1):

    if exp == 1:
        model_config = O2TBranchConfig(
            parametric=[64, 32, ],
            activation='relu',
            cov_mode='channel',
            cov_alpha=0.3,
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch='o2transform',
            cov_branch_output=32,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='Original-O2T-testing',
            nb_branch=2,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
        title = 'DTD-Testing-o2transform'
        running_config = get_running_config(title, model_config)
        running_config.comments = 'test mode 2 with dual branch '
    elif exp == 2:
        model_config = NoWVBranchConfig(
            parametric=[128],
            epsilon=1e-7,
            activation='relu',
            cov_mode='channel',
            cov_alpha=0.3,
            cov_beta=0.1,
            robust=False,
            normalization=False,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch_output=128,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='No-PV-2branch-128-128',
            nb_branch=2,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
        title = 'DTD-Testing-NO-PV'
        running_config = get_running_config(title, model_config)
        running_config.comments = 'Test branch 2 no WV concat with matrix again to reproduce our previous best'
    else:
        raise ValueError

    running_config.rescale_small = 256
    dtd_finetune_with_model(model_config, nb_epoch_finetune=4, running_config=running_config)


def test_minc_experiment(exp=1):
    pass


def vgg_baseline(exp=1):
    baseline_model_config = get_vgg_config()
    running_config = get_running_config('DTD-VGG6-baseline-withtop', baseline_model_config)
    dtd_finetune_with_model(baseline_model_config, 4, running_config)


if __name__ == '__main__':
    so_vgg_experiment(2)
    # vgg_baseline()
    # mpn_cov_baseline(1)
