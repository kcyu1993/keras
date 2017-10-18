"""
Define the Second-order training
"""
from kyu.configs.engine_configs.running import wrap_running_config
from kyu.configs.experiment_configs.running_configs import get_running_config_no_debug_withSGD
from kyu.configs.experiment_configs import simple_second_order_config as SOConfig
from kyu.datasets import get_dataset_by_name
from kyu.engine.utils.callbacks import TensorBoardWrapper
from kyu.experiment.data_train_utils import dtd_finetune_with_model, minc_finetune_with_model, sun_finetune_with_model, \
    mit_finetune_with_model, imagenet_finetune_with_model, data_finetune_with_model
from kyu.experiment.general_train import get_argparser
from kyu.utils import dict_utils

BRANCH_CHOICES = ['o2t_original', 'o2t_no_wv', 'o2t_wv_norm', 'o2t_wv_new_norm', '1x1_gsp']


def so_cnn_train(dataset, model_class, model_exp_fn, model_exp, nb_epoch_finetune=0, title='',
                 comments='', tf_dbg=False, tensorboard=False, **kwargs):
    """
    Generic training pipeline for argument parser

    Parameters
    ----------
    dataset : str       identifier of dataset
    model_class : str   model class
    model_exp_fn : func getting the model configuration based on the experiment
    model_exp : int     model experiment index
    nb_epoch_finetune : nb of epoch finetune running
    title : str         title of this training (once layer can be extended.
    comments : str      comments to be append in running config
    tf_dbg : bool        enter tfdbg mode
    tensorboard : bool  enable tensorboard logging.

    Returns
    -------

    """

    dataset = str(dataset).lower()
    model_config = model_exp_fn(model_exp)
    model_config.class_id = model_class

    running_config = get_running_config_no_debug_withSGD(
        title=title,
        model_config=model_config,
    )

    if tf_dbg:
        running_config.tf_debug = True
    running_config.tensorboard = tensorboard
    running_config.comments = comments
    wrap_running_config(config=running_config, **kwargs)
    data = get_dataset_by_name(dataset)
    if model_config.class_id == 'alexnet':
        model_config.load_weights = None

    data_finetune_with_model(data,
                             model_config=model_config,
                             nb_epoch_finetune=nb_epoch_finetune,
                             running_config=running_config)


def so_o2t_original(dataset, model_class, **kwargs):
    """ Get SO-VGG architecture with original o2t-branch """
    title = 'SO-{}_original'.format(str(model_class).upper())
    so_cnn_train(model_exp_fn=SOConfig.get_single_o2transform, model_class=model_class, dataset=dataset, title=title, **kwargs)


def so_o2t_no_wv(dataset, model_class, **kwargs):
    """ Get SO-CNN architecture with o2t no wv branch """
    title = 'SO-{}_noWV'.format(str(model_class).upper())
    so_cnn_train(model_exp_fn=SOConfig.get_no_wv_config, model_class=model_class, dataset=dataset, title=title, **kwargs)


def so_o2t_wv_with_norm(dataset, model_class, **kwargs):
    """ Get SO-CNN architecture with o2t wv with norm branch """
    title = 'SO-{}_normWV'.format(str(model_class).upper())
    so_cnn_train(model_exp_fn=SOConfig.get_wv_norm_config, model_class=model_class, dataset=dataset, title=title, **kwargs)


def so_o2t_wv_with_new_norm(dataset, model_class, **kwargs):
    """ Get SO-CNN architecture with o2t wv with new norm branch """
    title = 'SO-{}_normWV'.format(str(model_class).upper())
    so_cnn_train(model_exp_fn=SOConfig.get_new_wv_norm_general, model_class=model_class, dataset=dataset, title=title,
                 **kwargs)

def so_pv_equivelent(dataset, model_class, **kwargs):
    """ Get SO-CNN architecture with o2t wv with new norm branch """
    title = '{}-BN-Conv-GSP'.format(str(model_class).upper())
    so_cnn_train(model_exp_fn=SOConfig.get_pv_equivalent, model_class=model_class, dataset=dataset, title=title,
                 **kwargs)

if __name__ == '__main__':
    parser = get_argparser(description='SO-CNN architecture testing')
    parser.add_argument('-b', '--branch', help='second-order branch', default='o2t_original',
                        choices=BRANCH_CHOICES)

    args = parser.parse_args()
    branch = args.branch
    del args.branch

    if branch == BRANCH_CHOICES[0]:
        so_o2t_original(**vars(args))
    elif branch == BRANCH_CHOICES[1]:
        so_o2t_no_wv(**vars(args))
    elif branch == BRANCH_CHOICES[2]:
        so_o2t_wv_with_norm(**vars(args))
    elif branch == BRANCH_CHOICES[3]:
        so_o2t_wv_with_new_norm(**vars(args))
    elif branch == BRANCH_CHOICES[4]:
        so_pv_equivelent(**vars(args))
    else:
        raise NotImplementedError




