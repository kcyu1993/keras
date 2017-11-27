import h5py
import keras.backend as K
from keras.models import load_model
from kyu import DTD
from kyu.datasets.minc import Minc2500_v2, decode_minc_nnid
from kyu.experiment.general_train import get_data_generator_flags
from kyu.utils.imagenet_utils import deprocess_imagenet_for_visualization_without_channel_reverse, depreprocess_image_for_imagenet
from kyu.layers import get_custom_objects
from vis.utils import utils
from keras import activations

import numpy as np
from vis.visualization import visualize_cam, visualize_saliency, overlay
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def show_top_k_cam(model, layer_idx, result, top_k, img, original_img, label,
                   decode_fn=None, modifier=None, title=""):
    """
    Show the Top-k Class Activation Map for a single model, with a single image input.


    Notes
        one drawback is the duplication of calling this function, as the model is reloaded
        at each calling visualize_cam

    Parameters
    ----------
    model : keras model wrapped by keras-vis
    layer_idx : target layer to generate the CAM
    result : result format as generated.
    top_k : int, top k to be seen
    img : image input to the network
    original_img : original image
    label : np.array, as network label by generator
    decode_fn : decode the label
    modifier : None, guided and relu for visualize CAM
    title : title to be displayed

    Returns
    -------
    plt : matplotlib.pyplot for further process
    """
    if decode_fn is None:
        decode_fn = lambda x: x

    size = 6
    col = 3
    # determine the number of axis
    row = (top_k + 1) / col
    plt.rcParams['figure.figsize'] = (size * col, size * row)
    plt.figure()
    f, ax = plt.subplots(row, col)
    plt.suptitle("{} Modifier {}".format(title, modifier))
    ax[0][0].imshow(original_img)
    ax[0][0].set_title("Ground Truth: {}".format(decode_fn(np.argmax(label))))
    for i in range(1, top_k + 1):
        # plot the original image
        filter_indice = result[0][0][top_k - i]
        grads = visualize_cam(model, layer_idx, filter_indices=filter_indice,
                              seed_input=img, backprop_modifier=modifier)
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        ax[i / col][i % col].imshow(overlay(jet_heatmap, original_img))
        ax[i / col][i % col].set_title(
            "label:{}\n prob:{}".format(decode_fn(filter_indice), result[1][0][top_k - i]))

    return plt


def get_top_k_classification(model, seed_input, k=5):
    if np.ndim(seed_input) == 3:
        seed_input = np.expand_dims(seed_input, 0)
    predictions = model.predict(seed_input)
    #     print(predictions.shape)
    #     top4class_index = [i for i in np.argsort(preds[0])[-4:]]
    #     top4class_prob = [preds[0][i] for i in np.argsort(preds[0])[-4:]]

    top_k_preds = [list(i[-k:]) for i in np.argsort(predictions)]
    top_k_probs = [[predictions[j][i] for i in top_k_preds[j]] for j in range(predictions.shape[0])]
    #     print(top_k_preds)
    #     print(top_k_probs)
    return top_k_preds, top_k_probs