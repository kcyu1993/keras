from kyu.configs.engine_configs import ModelConfig

# TODO Rebuild the logic to have a FirstOrderModelConfig rather before that.


class FirstOrderModelConfig(ModelConfig):
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
        class_id = None
        model_id = None
        super(FirstOrderModelConfig, self).__init__()
        raise NotImplementedError


class AlexNetFirstOrderConfig(ModelConfig):
    # Define the compulsory stuff
    compulsory = ['nb_class', 'input_shape']
    optional = ['denses', 'input_tensor', 'weights', 'include_top', 'last_pooling', 'freeze_conv']

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
        class_id = 'alexnet'
        model_id = 'first_order'
        super(AlexNetFirstOrderConfig, self).__init__(class_id, model_id, **kwargs)
        self.__dict__.update(locals())
        self.batch_size = 128

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
        self.batch_size = 64


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
                 class_id='densenet121',
                 **kwargs
                 ):
        if class_id.find('densenet') < 0:
            raise ValueError("DenseNet model class id must contain densenet, got {}".
                             format(class_id))
        super(DenseNetFirstOrderConfig, self).__init__(class_id, 'first_order', **kwargs)
        self.__dict__.update(locals())


class ResNetFirstOrderConfig(ModelConfig):

    def __init__(self,
                 nb_class,
                 denses=[],
                 include_top=False,
                 input_shape=(224, 224, 3),
                 weights='imagenet',
                 input_tensor=None,
                 pooling=None,
                 last_avg=True,
                 pred_activation='softmax',
                 weight_decay=1e-4,
                 freeze_conv=False,
                 class_id='resnet50',
                 **kwargs
                 ):
        if class_id.find('resnet') < 0:
            raise ValueError("ResNet model must contains resnet {}".format(class_id))
        super(ResNetFirstOrderConfig, self).__init__(class_id, 'first_order', **kwargs)
        self.__dict__.update(locals())
