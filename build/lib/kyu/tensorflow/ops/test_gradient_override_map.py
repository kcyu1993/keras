"""
Test gradient override maps

"""
import tensorflow as tf
from tensorflow.python.framework import ops
import keras.backend as K
import numpy as np
# By the definition, we can use test gradient maps with the following informations.


# @tf.RegisterGradient("CustomGrad")
# def _const_mult_grad(unused_op, grad):
#     return 5.0 * grad
#
# g = tf.get_default_graph()
# with g.gradient_override_map({"Identity": "CustomGrad"}):
#     output = tf.identity(input, name='Identity')
#
def test_gradient_clipping_example_from_stackoverflow():

    @tf.RegisterGradient("CustomClipGrad")
    def _clip_grad(unused_op, grad):
        return tf.clip_by_value(grad, -0.1, 0.1)


    input = tf.Variable([3.0], dtype=tf.float32)
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": "CustomClipGrad"}):
        output_clip = tf.identity(input, name='Identity')
    grad_clip = tf.gradients(output_clip, input)


    output = tf.identity(input)
    grad = tf.gradients(output, input)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("with clipping : ", sess.run(grad_clip)[0])
        print("without clipping: ", sess.run(grad)[0])


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e+4))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc" : rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_func(func, grad=None):
    rnd_name = 'TfFuncGrad' + str(np.random.randint(0, 1e+4))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"TfFunc" : rnd_name}):
        return func




def test_gradient_override_with_keras():
    # @tf.RegisterGradient("CustomSqrtGrad")
    def _tf_sqrt_grad(op, grad):
        # if not isinstance(op, tf.ops.Operation):
        #     raise Exception
        x = op.inputs[0]
        dx = 0.5 / tf.sqrt(tf.abs(x) + 1e-5)
        res_grad = dx * grad
        return tf.clip_by_value(res_grad, -0.1, 0.1)

    def custom_sqrt(x):
        return tf.sqrt(tf.abs(x))

    def py_func_sqrt(x, name=None):
        with ops.op_scope([x], name, 'CustomSignedSqrt') as name:
            # sqr_x = tf.py_func(custom_sqrt, [x],[tf.float32], name=name)
            sqrt_x = py_func(custom_sqrt, [x], [tf.float32], name=name, grad=_tf_sqrt_grad)
            return sqrt_x

    def tf_func_sqrt(x, name=None):
        with ops.op_scope([x], name, 'TFCustomSignSqrt') as name:
            sqrt_x = tf_func(custom_sqrt, grad=_tf_sqrt_grad)(x)
            return sqrt_x

    with tf.Session() as sess:
        x = tf.constant([8., 2., 0.])
        # custom_y = py_func_sqrt(x)
        # custom_y = tf_func_sqrt(x)

        rnd_name = 'CustomGradientSqrt'
        # Register the gradient
        tf.RegisterGradient(rnd_name)(_tf_sqrt_grad)
        g = tf.get_default_graph()
        with g.gradient_override_map({'TfFunc': rnd_name}):
            with ops.op_scope([x], 'TfFunc') as name:
                custom_y = custom_sqrt(x)
            # custom_y = tf.sqrt(tf.abs(x), name='TfFunc')


        y = custom_sqrt(x)

        grad_y = tf.gradients(y, x)
        custom_grad_y = tf.gradients(custom_y, x)

        sess.run(tf.global_variables_initializer())

        print("Original y {} and its gradient {}".format(sess.run(y), sess.run(grad_y)))
        print('Custom y {} and its gradient {}'.format(sess.run(custom_y), sess.run(custom_grad_y)))

if __name__ == '__main__':
    test_gradient_override_with_keras()