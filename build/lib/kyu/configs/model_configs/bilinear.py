from kyu.configs.engine_configs import ModelConfig


class BilinearConfig(ModelConfig):

    def __init__(self,
                 nb_class,
                 input_shape,
                 class_id='vgg',
                 model_id='bilinear',
                 load_weights='imagenet'
                 ):
        super(BilinearConfig, self).__init__(class_id, model_id)
        self.__dict__.update(locals())