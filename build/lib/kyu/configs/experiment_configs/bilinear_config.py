from kyu.configs.model_configs.bilinear import BilinearConfig


def get_bilinear_baseline_exp(exp=1):

    if exp == 1:
        return BilinearConfig(
            nb_class=10,
            input_shape=(224,224,3),
            load_weights='imagenet'
        )
    else:
        raise NotImplementedError

