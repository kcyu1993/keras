from kyu.configs.experiment_configs.bilinear_config import get_bilinear_baseline_exp
from kyu.experiment.so_train import so_cnn_train


def bilinear_baseline(model_class, **kwargs):
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
