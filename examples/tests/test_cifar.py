import pytest
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from keras.applications.resnet50 import ResCovNet50CIFAR
from keras.utils.test_utils import image_classification, keras_test
from kyu.models.cifar import cifar_fitnet_v2, cifar_fitnet_v1, cifar_fitnet_v3, model_snd

nb_class = 10
input_shape = (3, 32, 32)


# @keras_test
# def test_ResCovNetCIFAR_nonparam():
#     for i in range(4, 7):
#         model = ResCovNet50CIFAR([], nb_class=10, mode=i)
#         image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
#     # assert 0
#
#
# @keras_test
# def test_ResCovNetCIFAR_param():
#     param = [100, 100]
#     for i in range(4, 7):
#         model = ResCovNet50CIFAR(param, nb_class=10, mode=i)
#         image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
#     # assert 0
#
#
# @keras_test
# def test_fitnet_param():
#     paras = [[100, ], [50, ], [100, 50]]
#     for para in paras:
#         model = cifar_fitnet_v1(True, para)
#         image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
#     model = cifar_fitnet_v1(False)
#     image_classification(model, nb_class=10, input_shape=input_shape, fit=False)


# @keras_test
# def test_fitnet_v3_params():
#     params = [[], [50], [100], [100, 50], [16, 8], [32, 16], [16, 32]]
#     nb_epoch = 200
#     cov_outputs = [500, 100, 50, 10]
#     for cov_output in cov_outputs:
#         for param in params:
#             print("Run routine 13 nb epoch {} param mode {}".format(nb_epoch, param))
#             for mode in range(0,10):
#                 model = cifar_fitnet_v3(parametrics=param, epsilon=0, mode=mode,
#                                         nb_classes=nb_class, dropout=False,
#                                         cov_mode='o2transform', cov_branch_output=cov_output)
#                 image_classification(model, nb_class=nb_class, input_shape=input_shape, fit=False)


@keras_test
def test_log_model():
    model = model_snd(parametrics=[])
    image_classification(model, nb_class=nb_class, input_shape=input_shape, fit=True)


if __name__ == '__main__':

    pytest.main([__file__])

