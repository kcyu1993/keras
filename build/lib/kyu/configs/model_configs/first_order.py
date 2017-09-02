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


class DenseNetFirstOrderConfig(ModelConfig):

    def __init__(self,
                 nb_class,
                 input_shape=(224, 224, 3),
                 nb_dense_block=4,
                 growth_rate=32,
                 nb_filter=64,
                 reduction=0.0,
                 dropout_rate=0.0,
                 weight_decay=1e-4,
                 weights_path='imagenet',
                 freeze_conv=False,
                 last_pooling=True,
                 **kwargs
                 ):
        super(DenseNetFirstOrderConfig, self).__init__('densenet121', 'first_order', **kwargs)
        self.__dict__.update(locals())
