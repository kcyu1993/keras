"""
Test the CAM with MINC-2500 model

"""
import cv2
import numpy as np
import h5py
import keras.backend as K
from keras.models import load_model
from kyu import DTD

from kyu.datasets.minc import Minc2500_v2
from kyu.experiment.general_train import get_data_generator_flags
from kyu.utils.imagenet_utils import deprocess_imagenet_for_visualization_without_channel_reverse
from kyu.layers.secondstat import get_custom_objects

VGG_GSP_DTD_PATH = '/Users/kcyu/mount/cvlabdata3/cvkyu/so_updated_record/output/' \
                   'run/cls/DTD/vgg16/VGG16-BN-Conv-GSPBN-1x1_2048-GSP-useGamme_True' \
                   '-1_branch_2017-10-12T18:17:15/keras_model.41-3.45.hdf5'
VGG_GSP_DTD_WEIGHT = '/Users/kcyu/mount/cvlabdata3/cvkyu/so_updated_record/output/' \
                   'run/cls/DTD/vgg16/VGG16-BN-Conv-GSPBN-1x1_2048-GSP-useGamme_True' \
                   '-1_branch_2017-10-12T18:17:15/keras_model.41-3.45.hdf5.weight'


def load_model_weights(model, weights_path):
    """

    Parameters
    ----------
    model
    weights_path

    Returns
    -------

    """
    print('Loading model.')
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        model.layers[k].trainable = False
    f.close()
    print('Model loaded.')
    return model


def get_output_layer(model, layer_name):
    """

    Parameters
    ----------
    model
    layer_name

    Returns
    -------

    """
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def generate_example(dataset, target_size=(224,224)):
    """

    Returns
    -------
    data
    """
    if dataset == 'minc':
        data = Minc2500_v2(dirpath='/Users/kcyu/mount/cvdata/cvlab/datasets_kyu/minc-2500/')
    elif dataset == 'dtd':
        data = DTD('/Users/kcyu/mount/cvlabdata/dtd', image_dir='images')
    else:
        return

    # Get the generator
    gen = get_data_generator_flags(str('VGG').lower(),
                                   target_size,
                                   data,
                                   'test')

    test = data.get_test(batch_size=1, image_data_generator=gen)
    return test


def test_dtd_CAM_for_trail():
    """
    Trail

    Returns
    -------

    """

    # load model
    model = load_model(VGG_GSP_DTD_PATH, get_custom_objects())
    model.load_weights(VGG_GSP_DTD_WEIGHT)

    final_conv_layer = model.layers[-3]

    # prepare the prediction and data
    test_gen = generate_example('dtd')

    # number of plots
    num = 16

    for ind in enumerate(range(num)):
        output_path = '/Users/kcyu/mount/plots/dtd_test_{}.png'.format(ind)
        img, label = next(test_gen)
        # load related stuff
        original_img = deprocess_imagenet_for_visualization_without_channel_reverse(img[0,:,:,:])
        visualize_class_activation_map(model, final_conv_layer, original_img, img, output_path)


def visualize_class_activation_map(model, final_conv_layer, original_img, img, output_path, threshold=0.3):
    """
    Visualize CAM with model, original img, image and output path.

    Version 1.0
        Only support manual calibration
    Parameters
    ----------
    model : keras model with correct weights
    final_conv_layer : layer to be plotted
    original_img : original image
    img : the actual image in neural network
    output_path : output path

    Returns
    -------

    """

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    height, width = img.shape[1], img.shape[2]

    get_output = K.function([model.layers[0].input, K.learning_phase()], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img, 1])
    conv_output = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for i, w in enumerate(class_weights[:, 1]):
        cam += w * conv_output[:, :, i]
    # print("predictions" + predictions)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < threshold)] = 0
    img = heatmap * 0.5 + original_img
    cv2.imwrite(output_path, img)
    return predictions

if __name__ == '__main__':
    # valid the installation
    print(cv2.__version__)