from kyu.configs.model_configs.second_order import DCovConfig


class BilinearConfig(DCovConfig):

    def __init__(self,
                 nb_class,
                 input_shape,
                 class_id='vgg',
                 model_id='bilinear',
                 load_weights='imagenet',
                 last_conv_kernel=[],
                 **kwargs
                 ):

        super(BilinearConfig, self).__init__(class_id, model_id, **kwargs)
        self.__dict__.update(locals())