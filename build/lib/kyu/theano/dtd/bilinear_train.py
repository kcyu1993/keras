from kyu.configs.model_configs.bilinear import BilinearConfig
from kyu.theano.dtd.new_train import get_running_config, finetune_with_model


def bilinear_baseline(exp=1):

    bilinear_config = BilinearConfig(nb_class=67, input_shape=(224,224,3))

    running_config = get_running_config('Bilinear-baseline-noclip', bilinear_config)

    # running_config.optimizer = get_test_optimizer()
    running_config.comments = 'No clipping'

    finetune_with_model(bilinear_config, 0, running_config)

if __name__ == '__main__':
    bilinear_baseline(1)