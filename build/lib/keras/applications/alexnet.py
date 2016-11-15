from __future__ import absolute_import

import numpy as np
from scipy.misc import imread, imresize

from keras.applications.convnets_custom import crosschannelnormalization, \
    splittensor, Softmax4D
from keras.layers import Flatten, Dense, Dropout, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import SGD
# from keras.utils.image_tool import synset_to_dfs_ids
from keras.utils.model_utils import convnet_alexnet

def convnet(network, weights_path=None, heatmap=False,
            trainable=None):
    """
    Returns a keras model for a CNN.
    BEWARE !! : Since the different convnets have been trained in different settings, they don't take
    data of the same shape. You should change the arguments of preprocess_image_batch for each CNN :
    * For alexnet, the data are of shape (227,227), and the colors in the RGB order (default)

    It can also be used to look at the hidden layers of the model.
    It can be used that way :
    # >>> im = preprocess_image_batch(['cat.jpg'])
    # >>> # Test pretrained model
    # >>> model = convnet('vgg_16', 'weights/vgg16_weights.h5')
    # >>> sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # >>> model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # >>> out = model.predict(im)
    Parameters
    --------------
    network: str
        The type of network chosen. For the moment, can be 'vgg_16' or 'vgg_19'
    weights_path: str
        Location of the pre-trained model. If not given, the model will be trained
    heatmap: bool
        Says wether the fully connected layers are transformed into Convolution2D layers,
        to produce a heatmap instead of a
    Returns
    ---------------
    model:
        The keras model for this convnet
    output_dict:
        Dict of feature layers, asked for in output_layers.
    """


    # Select the network
    # if network == 'vgg_16':
    #     convnet_init = VGG_16
    # elif network == 'vgg_19':
    #     convnet_init = VGG_19
    if network == 'alexnet':
        convnet_init = alexnet
    convnet = convnet_init(weights_path, heatmap=False)

    if not heatmap:
        return convnet
    else:
        convnet_heatmap = convnet_init(heatmap=True)

        for layer in convnet_heatmap.layers:
            if layer.name.startswith("conv"):
                orig_layer = convnet.get_layer(layer.name)
                layer.set_weights(orig_layer.get_weights())
            elif layer.name.startswith("dense"):
                orig_layer = convnet.get_layer(layer.name)
                W,b = orig_layer.get_weights()
                n_filter,previous_filter,ax1,ax2 = layer.get_weights()[0].shape
                new_W = W.reshape((previous_filter,ax1,ax2,n_filter))
                new_W = new_W.transpose((3,0,1,2))
                new_W = new_W[:,:,::-1,::-1]
                layer.set_weights([new_W,b])
        return convnet_heatmap

    return model


def alexnet(weights_path=None, heatmap=False):
    if heatmap:
        inputs = Input(shape=(3,None,None))
    else:
        inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    if heatmap:
        dense_1 = Convolution2D(4096,6,6,activation="relu",name="dense_1")(dense_1)
        dense_2 = Convolution2D(4096,1,1,activation="relu",name="dense_2")(dense_1)
        dense_3 = Convolution2D(1000, 1,1,name="dense_3")(dense_2)
        prediction = Softmax4D(axis=1,name="softmax")(dense_3)
    else:
        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000,name='dense_3')(dense_3)
        prediction = Activation("softmax",name="softmax")(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)

    return model


def alexnet_top(weights_path=None, heatmap=False, input_shape=(3,227,227)):
    """
    Return the alex net without dense layer.
    :param weights_path:
    :param heatmap:
    :param input_shape:     should not be modified with weight_path
    :return:
    """
    if heatmap:
        inputs = Input(shape=(3,None,None))
    else:
        inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    return dense_1


def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if color_mode=="bgr":
            img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:,(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                      ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch


if __name__ == "__main__":
    ### Here is a script to compute the heatmap of the dog synsets.
    ## We find the synsets corresponding to dogs on ImageNet website
    s = "n02084071"
    # ids = synset_to_dfs_ids(s)
    # # Most of the synsets are not in the subset of the synsets used in ImageNet recognition task.
    # ids = np.array([id_ for id_ in ids if id_ is not None])
    #
    # im = preprocess_image_batch(['examples/dog.jpg'], color_mode="rgb")
    #
    # # Test pretrained model
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model = convnet('alexnet',weights_path=convnet_alexnet(), heatmap=True)
    # model.compile(optimizer=sgd, loss='mse')
    #
    #
    # out = model.predict(im)
    # heatmap = out[0,ids,:,:].sum(axis=0)