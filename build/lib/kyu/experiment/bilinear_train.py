from kyu.configs.experiment_configs.bilinear_config import get_bilinear_baseline_exp
from kyu.experiment.general_train import get_argparser
from kyu.experiment.so_train import so_cnn_train


def bilinear_train(model_class, **kwargs):
    """
    B-CNN baseline testing

    Parameters
    ----------
    model_class
    kwargs

    Returns
    -------

    """
    return so_cnn_train(model_class=model_class, model_exp_fn=get_bilinear_baseline_exp,
                        title='B-CNN-{}-baseline'.format(str(model_class).upper()),
                        **kwargs)


if __name__ == '__main__':
    parser = get_argparser(description='Bilinear Train with different dataset and model settings ')
    bilinear_train(**vars(parser.parse_args()))
