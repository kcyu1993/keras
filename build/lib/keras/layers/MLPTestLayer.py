# from __future__ import absolute_import
#
# from .. import backend as K
# from .. import activations, initializations, regularizers, constraints
# from ..engine import Layer, InputSpec
# from ..utils.np_utils import conv_output_length, conv_input_length
#
# class MLPLayer(Layer):
#     def __init__(self, W_init, b_init, activation, **kwargs):
#
#         # Set the according parameters in this layer
#         self.init = initializations.get('uniform',dim_ordering='th')
#         self.activation = activations.get(activation)
#         self.W_init = W_init
#         self.b_init = b_init
#         self.in_dim, self.out_dim = W_init.shape
#
#         super(MLPLayer,self).__init__(**kwargs)
#
#     def build(self, input_shape):
#
#         input_dim =
