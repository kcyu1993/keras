"""
Define the Second-order training
"""
from kyu.configs.experiment_configs.running_configs import get_running_config_no_debug_withSGD
from kyu.configs.experiment_configs.simple_second_order import get_single_o2transform, get_no_wv_config
from kyu.experiment.general_train import get_argparser
from kyu.experiment.minc_utils import minc_finetune_with_model
from kyu.experiment.dtd_utils import dtd_finetune_with_model

BRANCH_CHOICES = ['o2t_original', 'o2t_no_wv']


def so_cnn_train(dataset, model_config, running_config, nb_epoch_finetune=0):
    if dataset == 'dtd':
        dtd_finetune_with_model(model_config, nb_epoch_finetune, running_config)
    elif dataset == 'minc2500' or dataset == 'minc-2500':
        minc_finetune_with_model(model_config, nb_epoch_finetune, running_config)


def so_o2t_original(model_class, dataset, model_exp=1, nb_epoch_finetune=0):
    """ Get SO-VGG architecture with original o2t-branch """
    model_config = get_single_o2transform(model_exp)
    model_config.class_id = model_class
    running_config = get_running_config_no_debug_withSGD(
        title='SO-{} original {}'.format(str(model_class).upper(),dataset),
        model_config=model_config
    )

    so_cnn_train(dataset, model_config, running_config, nb_epoch_finetune)


def so_o2t_no_wv(model_class, dataset, model_exp=1, nb_epoch_finetune=0):

    model_config = get_no_wv_config(model_exp)
    model_config.class_id = model_class
    running_config = get_running_config_no_debug_withSGD(
        title='SO-{} no wv {}'.format(str(model_class).upper(), dataset),
        model_config=model_config
    )
    so_cnn_train(dataset, model_config, running_config, nb_epoch_finetune)


if __name__ == '__main__':
    parser = get_argparser(description='SO-CNN architecture testing')
    parser.add_argument('-m','--model_class', help='model class should be in vgg, resnet', default='vgg', type=str)
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




