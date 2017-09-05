"""
Baseline for first order tests

"""
from kyu.configs.experiment_configs.first_order import get_fo_vgg_exp, get_fo_dense_exp
from kyu.experiment.general_train import get_argparser
from kyu.experiment.so_train import so_cnn_train


def baseline_first_order_train(model_class, **kwargs):

    if str(model_class).lower() == 'vgg':
        model_exp_fn = get_fo_vgg_exp
    elif str(model_class).lower() == 'densenet121':
        model_exp_fn = get_fo_dense_exp
    else:
        raise NotImplementedError

    return so_cnn_train(model_class=model_class,
                        model_exp_fn=model_exp_fn,
                        title="FO-{}-baseline".format(str(model_class).upper()),
                        **kwargs)


if __name__ == '__main__':
    parser = get_argparser(description='FO Train with different settings ')
    args = parser.parse_args()

    # Model class
    baseline_first_order_train(**vars(args))
