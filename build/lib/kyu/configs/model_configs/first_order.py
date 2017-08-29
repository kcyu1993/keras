from kyu.configs.engine_configs import ModelConfig


class VggFirstOrderConfig(ModelConfig):

    def __init__(self,
                 nb_classes, input_shape,
                 denses=[],
                 input_tensor=None,
                 weights='imagenet',
                 include_top=True,
                 freeze_conv=False,
                 last_pooling=True,
                 **kwargs
                 ):
        class_id = 'vgg'
        model_id = 'first_order'
        super(VggFirstOrderConfig, self).__init__(class_id, model_id, **kwargs)
        self.__dict__.update(locals())
