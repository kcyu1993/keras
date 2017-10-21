"""
Train a SVM out of CNN model

"""
import argparse

import numpy
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

import keras.backend as K
from keras.activations import linear
from keras.engine import Model
from keras.layers import Dense, initializers
from keras.losses import categorical_hinge
from keras.models import load_model
from kyu import get_custom_objects, get_dataset_by_name
from kyu.configs.experiment_configs.running_configs import get_running_config_no_debug_withSGD
from kyu.experiment.general_train import get_data_generator_flags
from kyu.utils.dict_utils import update_source_dict_by_given_kwargs
from kyu.utils.image import ImageIterator

CAR_WEIGHTS='/home/kyu/cvkyu/secondstat_final/output/run/cls/StanfordCar/' \
            'vgg16/SO-VGG16_normWVBN-Cov-PV2048_final-BN-1_branch/' \
            'so-baseline_2017-10-19T09:17:55/keras_model.56-0.51.hdf5'

CUB_WEIGHTS='/cvlabdata3/kyu/cvkyu/secondstat_final/output/run/cls/CUB2011/vgg16/SO-VGG16_normWVBN-Cov-PV2048_final-BN-448-1_branch/so-baseline_2017-10-19T07:27:41/keras_model.88-0.93.hdf5'

MIT_WEIGHTS='/home/kyu/cvkyu/secondstat_final/output/run/cls/MitIndoor/vgg/' \
            'SO-VGG_normWVBN-Cov-PV2048_final-BN-448-1_branch/' \
            'final-model-with-448-reception_2017-10-19T14:36:23/keras_model.30-1.04.hdf5'


def get_argparser(description='default'):
    """
    Define the get argument parser default, given the description of the task

    Parameters
    ----------
    description: str  description of the task.

    Returns
    -------
    parser with all defaults

    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset name: support dtd, minc2500')
    parser.add_argument('-c', '--comments', help='comments if any', default='', type=str)

    parser.add_argument('-debug', '--debug', help='set debug flag', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    parser.add_argument('-lw', '--load_weights', help='set for training from scratch', action='store_true',
                        dest='load_weights')
    parser.set_defaults(load_weights=False)
    return parser


def obtain_intermediate_layer_output(model, target_layer_name_key=None, target_layer_index=-1,
                                     **kwargs):
    """
    Define the intermediate layer output obtaining to train the SVM.

    Parameters
    ----------
    model
    target_layer_name_key : higher priority
    target_layer_index : the index of the layer
    kwargs :

    Returns
    -------

    """
    layer = None
    if target_layer_index:
        layer = model.get_layer(index=target_layer_index)

    if target_layer_name_key:
        for l in model.layers:
            if target_layer_name_key in l:
                layer = model.get_layer(name=l)
    if layer is None:
        raise ValueError("Layer not proper defined")

    intermediate_model = Model(inputs=model.input,
                               outputs=layer.output,
                               name=model.name + layer.name)

    return intermediate_model


def obtain_dataset_cnn_features(model, data, **kwargs):
    """
    Obtain the CNN features based on model

    Parameters
    ----------
    model
    data : ImageIterator or nparray
    kwargs

    Returns
    -------

    """
    if isinstance(data, numpy.ndarray):
        output = model.predict(data, **kwargs)
    elif isinstance(data, ImageIterator):
        output = model.predict_generator(data, **kwargs)
    else:
        raise ValueError("Not supported data type {}".format(data))

    return output


def generate_compact_label_tensor(label):
    """

    Parameters
    ----------
    label : ndarray in shape (None, num_lab)

    Returns
    -------
    label: ndarray in shape (None,)
    """
    if not isinstance(label, numpy.ndarray):
        print("Only accept ndarray")
        return None
    if not len(label.shape) == 2:
        print("Only support ndarray with dim 2")
        return None
    return numpy.sum(label * numpy.asanyarray(range(label.shape[1])), axis=1)


def svc_svm_train(train, test):
    """
    Train SVM via SKLearn SVC
    Parameters
    ----------
    train
    test

    Returns
    -------

    """
    print("Start training SVM ...")
    svcClf = SVC(C=1.0, kernel='linear', cache_size=3000)
    compact_train_label = generate_compact_label_tensor(train[1])
    compact_test_label = generate_compact_label_tensor(test[1])

    svcClf.fit(train[0], compact_train_label)

    print("Finish training SVM ... \n Start prediction ...")
    pred_testlabel = svcClf.predict(test[0])

    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num)
                    if compact_test_label[i] == pred_testlabel[i]])\
               / float(num)
    print("CNN-SVM train with accuracy {}".format(accuracy))

    return svcClf


def svm_train_sklearn(load_weights, dataset, **kwargs):
    """
    Train the SVM with SKlearn rather than the keras model

    Parameters
    ----------
    load_weights
    dataset
    kwargs

    Returns
    -------

    """

    # Define the model as similar
    dataset = 'cub'
    data = get_dataset_by_name(dataset)

    # TO train the SVM, use the full image crop rather than random croping
    data.train_image_gen_configs = update_source_dict_by_given_kwargs(
        data.train_image_gen_configs, random_crop=False, rescaleshortedgeto=448)
    data.valid_image_gen_configs = update_source_dict_by_given_kwargs(
        rescaleshortedgeto=448
    )

    model_path = load_weights if load_weights else CUB_WEIGHTS
    model = load_model(model_path, custom_objects=get_custom_objects())

    # obtain the intermediate model
    model = obtain_intermediate_layer_output(model, target_layer_index=-2)

    input_shape = K.int_shape(model.input)
    target_size = input_shape[1:3] if K.image_data_format() == 'channels_last' else input_shape[2:4]
    train_image_gen = get_data_generator_flags('vgg16', target_size, data, mode='train')
    valid_image_gen = get_data_generator_flags('vgg16', target_size, data, mode='valid')

    train = data.get_train(batch_size=data.train_n,
                           target_size=target_size,
                           image_data_generator=train_image_gen)

    valid = data.get_test(batch_size=data.valid_n,
                          target_size=target_size,
                          image_data_generator=valid_image_gen)
    valid_data = valid.next()
    train_data = train.next()
    print("Get CNN image features ... ")
    train_cnn_features = obtain_dataset_cnn_features(model, train_data[0])
    valid_cnn_features = obtain_dataset_cnn_features(model, valid_data[0])

    svc = svc_svm_train([train_cnn_features, train_data[1]], [valid_cnn_features, valid_data[1]])



# TODO Finish this to see any potential improvement on the Fine-grained datasets.

def train_svm(load_weights, dataset, **kwargs):
    # Load the model
    dataset = 'cub'
    model_path = load_weights if load_weights else CUB_WEIGHTS
    model = load_model(model_path, custom_objects=get_custom_objects())

    # Assigned to linear activation to the prediction layer
    dense_layer = model.layers[-1]
    if isinstance(dense_layer, Dense):

        dense_layer.activation = linear
        dense_layer.kernel = dense_layer.add_weight(
            shape=K.int_shape(dense_layer.kernel),
            initializer=dense_layer.kernel_initializer,
            name='kernel',
            regularizer=dense_layer.kernel_regularizer,
            constraint=dense_layer.kernel_constraint)
        if dense_layer.use_bias:
            dense_layer.use_bias = dense_layer.add_weight(
                shape=K.int_shape(dense_layer.bias),
                initializer=dense_layer.bias_initializer,
                name='kernel',
                regularizer=dense_layer.bias_regularizer,
                constraint=dense_layer.bias_constraint)
    for layer in model.layers[:-1]:
        layer.trainable = False

    data = get_dataset_by_name(dataset)

    running_config = get_running_config_no_debug_withSGD(
        title='SVM test',
    )

    # Get data generator
    if running_config.train_image_gen_configs:
        data.train_image_gen_configs = running_config.train_image_gen_configs
    if running_config.valid_image_gen_configs:
        data.valid_image_gen_configs = running_config.valid_image_gen_configs

    input_shape = K.int_shape(model.input)
    target_size = input_shape[1:3] if K.image_data_format() == 'channels_last' else input_shape[2:4]

    train_image_gen = get_data_generator_flags('vgg16', target_size, data, mode='train')
    valid_image_gen = get_data_generator_flags('vgg16', target_size, data, mode='valid')

    # Build the id.

    model.compile('adadelta', categorical_hinge, ['acc'])

    train = data.get_train(batch_size=running_config.batch_size,
                           target_size=target_size,
                           image_data_generator=train_image_gen)

    valid = data.get_test(batch_size=running_config.batch_size,
                          target_size=target_size,
                          image_data_generator=valid_image_gen)
    steps_per_epoch = min(train.n / train.batch_size, running_config.train_nb_batch_per_epoch)
    val_steps_per_epoch = min(valid.n / valid.batch_size if valid is not None else 0,
                              running_config.val_nb_batch_per_epoch)

    model.fit_generator(train, steps_per_epoch=steps_per_epoch, epochs=5, verbose=1,
                        validation_data=valid, validation_steps=val_steps_per_epoch, )

if __name__ == '__main__':
    parser = get_argparser('SVM training')

    args = parser.parse_args()
    train_svm(**vars(args))
