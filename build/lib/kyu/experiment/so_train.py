"""
Define the Second-order training
"""
from kyu.configs.engine_configs.running import wrap_running_config
from kyu.configs.experiment_configs.running_configs import get_running_config_no_debug_withSGD
from kyu.configs.experiment_configs.simple_second_order_config import get_single_o2transform, get_no_wv_config
from kyu.experiment.data_train_utils import dtd_finetune_with_model, minc_finetune_with_model, sun_finetune_with_model, \
    mit_finetune_with_model
from kyu.experiment.general_train import get_argparser

BRANCH_CHOICES = ['o2t_original', 'o2t_no_wv']


def so_cnn_train(dataset, model_class, model_exp_fn, model_exp, nb_epoch_finetune=0, title='',
                 comments='', debug=False, tensorboard=False, **kwargs):
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
    debug : bool        enter tfdbg mode
    tensorboard : bool  enable tensorboard logging.

    Returns
    -------

    """
    dataset = str(dataset).lower()
    model_config = model_exp_fn(model_exp)
    model_config.class_id = model_class
    running_config = get_running_config_no_debug_withSGD(
        title=title,
        model_config=model_config
    )

    if debug:
        running_config.tf_debug = True
    running_config.tensorboard = tensorboard
    running_config.comments = comments
    wrap_running_config(config=running_config, **kwargs)

    if dataset == 'dtd':
        dtd_finetune_with_model(model_config, nb_epoch_finetune, running_config)
    elif dataset == 'minc2500' or dataset == 'minc-2500':
        minc_finetune_with_model(model_config, nb_epoch_finetune, running_config)
    elif dataset == 'sun':
        sun_finetune_with_model(
            model_config=model_config,
            nb_epoch_finetune=nb_epoch_finetune,
            running_config=running_config)
    elif dataset in ['mit', 'mit_indoor', 'mitindoor']:
        mit_finetune_with_model(
            model_config=model_config,
            nb_epoch_finetune=nb_epoch_finetune,
            running_config=running_config
        )
    else:
        raise ValueError("Dataset not supported {}".format(dataset))


def so_o2t_original(dataset, model_class, **kwargs):
    """ Get SO-VGG architecture with original o2t-branch """
    title = 'SO-{}_original'.format(str(model_class).upper())
    so_cnn_train(model_exp_fn=get_single_o2transform, model_class=model_class, dataset=dataset, title=title, **kwargs)


def so_o2t_no_wv(dataset, model_class, **kwargs):
    """ Get SO-CNN architecture with o2t no wv branch """
    title = 'SO-{}_noWV'.format(str(model_class).upper())
    so_cnn_train(model_exp_fn=get_no_wv_config, model_class=model_class, dataset=dataset, title=title, **kwargs)


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
    else:
        raise NotImplementedError




