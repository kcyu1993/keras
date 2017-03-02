# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



"""
Make sure it could be run [check]
Add the keras model into such process to make it trainable in TF framework

Use this CIFAR example as general framework to feed model
"""

from datetime import datetime
import time
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

from kyu.tensorflow.cifar import cifar10


# Keras models
import keras
import keras.backend as K
if K.backend() == 'tensorflow':
    K.set_image_dim_ordering('tf')

from kyu.models.cifar import cifar_fitnet_v4
# from kyu.tensorflow.cifar.cifar10_models import fitnet_inference
# import kyu.tensorflow.cifar.cifar10_models as cifar10
# from keras.objectives import categorical_crossentropy


from kyu.tensorflow.cifar.cifar10_slim import fitnet_slim, simple_slim_model, simple_second_model, simple_log_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/kyu/cvkyu/tensorboard/cifar10/simpleSecondModel/DCov_Log_2',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# tf.app.flags.DEFINE_boolean('debug', True,'debug info')
K.set_learning_phase(1)

# inference = cifar_fitnet_v3(input_shape=(24, 24, 3), dropout=False, last_softmax=False)
inference = simple_second_model
# inference = simple_log_model
# inference = simple_slim_model
# inference = cifar10.inference


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # images, labels = cifar10.distorted_inputs()
        images, labels = cifar10.inputs(False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # logits = cifar10.inference(images)

        # Use slim to build model
        logits = inference(images)

        # Calculate loss.
        loss = cifar10.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))
                if self._step % 5000 == 0 and self._step != 0:
                    # Save the model
                    pass
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                save_checkpoint_secs=1800,
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=FLAGS.log_device_placement),

        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
        # with tf.Session(
        #         config=tf.ConfigProto(
        #             allow_soft_placement=True,
        #             log_device_placement=FLAGS.log_device_placement
        #
        #         )) as mon_sess:
        #     mon_sess = tf_debug.LocalCLIDebugWrapperSession(mon_sess)
        #     mon_sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
        #     mon_sess.run(train_op)


def train_keras():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # images, labels = cifar10.distorted_inputs()
        images, labels = cifar10.inputs(False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # logits = cifar10.inference(images)

        # Calculate loss.

        # Define keras part
        print('keras version ' + keras.__version__)
        K.set_learning_phase(1)
        model = cifar_fitnet_v4(input_shape=(24, 24, 3), dropout=True, last_softmax=False)

        logits = model(images)

        # loss = tf.reduce_mean(K.categorical_crossentropy(logits, labels, from_logits=True))
        # test_image = tf.placeholder(tf.float32, shape=(None, 24, 24, 3))
        # logits = cifar_fitnet_v1_test(test_image)

        # logits = cifar_fitnet_v1_test(images)

        # logits = cifar10.fitnet_inference(images)

        loss = cifar10.loss(logits, labels)
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                save_checkpoint_secs=1800,  # Save the model every 1800 seconds
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            # mon_sess = tf_debug.LocalCLIDebugWrapperSession(mon_sess)
            # mon_sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
    # train_keras()

if __name__ == '__main__':
    tf.app.run()
