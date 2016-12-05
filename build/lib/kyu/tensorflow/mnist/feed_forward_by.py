import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data

FLAGS = None


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                          mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    """
    Call each time before the network trained, basically each epoch.

    Parameters
    ----------
    data_set
    images_pl
    labels_pl

    Returns
    -------

    """
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }
    return feed_dict


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size