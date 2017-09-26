"""
Train with Prototype network

"""
from kyu.configs.engine_configs.running import wrap_running_config
from kyu.configs.experiment_configs import mpn_config
from kyu.configs.experiment_configs.running_configs import get_running_config_no_debug_withSGD
from kyu.datasets import get_dataset_by_name
from kyu.experiment.general_train import get_argparser, finetune_with_model_and_data_without_config
from kyu.experiment.so_train import so_cnn_train
from kyu.models.prototype_norm_so_cnn import ResNet50_so_prototype

BRANCH_CHOICES = ['mpn_baseline', 'multi_mpn']


def prototype_train(model_class, dataset, nb_epoch_finetune=0, title='',
                    comments='', tf_dbg=False, tensorboard=False, **kwargs):
    """
    Prototype_train similar to SO-CNN Train, for a generalized model testing
    Parameters
    ----------
    model_class
    dataset
    model_exp_fn
    model_exp
    nb_epoch_finetune
    title
    comments
    tf_dbg
    tensorboard
    kwargs

    Returns
    -------

    """
    dataset = str(dataset).lower()
    running_config = running_config = get_running_config_no_debug_withSGD(
        title=title,
    )
    if str(model_class).lower().find('resnet') >= 0:
        ResNet50_so_prototype()
    if tf_dbg:
        running_config.tf_debug = True
    running_config.tensorboard = tensorboard
    running_config.comments = comments
    wrap_running_config(config=running_config, **kwargs)
    data = get_dataset_by_name(dataset)
    # finetune_with_model_and_data_without_config(data,
    #                                             model=model
    #                          nb_epoch_finetune=nb_epoch_finetune,
    #                          running_config=running_config)


if __name__ == '__main__':
    parser = get_argparser(description='Prototype Train with different dataset and model settings ')
    parser.add_argument('-b', '--branch', default='Prototype', choices=BRANCH_CHOICES)

    args = parser.parse_args()
    branch = args.branch
    del args.branch

    if branch == BRANCH_CHOICES[0]:
        prototype_train(**vars(args))
    elif branch == BRANCH_CHOICES[1]:
        prototype_train(**vars(args))
    else:
        raise ValueError


