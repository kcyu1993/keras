from kyu.configs.model_configs import BilinearConfig


def get_bilinear_baseline_exp(exp=1):

    if exp == 1:
        return BilinearConfig(
            nb_class=10,
            input_shape=(224,224,3),
            load_weights='imagenet',
            name='BCNN-Baseline',
        )
    elif exp == 2:
        return BilinearConfig(
            nb_class=10,
            input_shape=(224, 224, 3),
            last_conv_kernel=[256],
            load_weights='imagenet',
            name='BCNN-Baseline',
        )
    else:
        raise NotImplementedError


def get_bilinear_so_structure_exp(exp=1):
    if exp == 2:
        return BilinearConfig(
            nb_class=10,
            input_shape=(224, 224, 3),
            load_weights='imagenet',
            name='BCNN-Baseline',
        )