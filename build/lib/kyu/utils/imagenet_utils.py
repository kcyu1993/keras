from keras import backend as K


def preprocess_image_for_imagenet_of_densenet(img):
    """
    Data preprocess for densenet.
    Subtract mean pixel and multiple by scaling constant
    Reference: https://github.com/shicai/DenseNet-Caffe

    Parameters
    ----------
    img : ndarray

    Returns
    -------
    img: ndarray with same shape
    """
    data_format = K.image_data_format()
    assert data_format in {'channel_last', 'channel_first'}
    im = img
    if data_format == 'channel_last':
        im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
        im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
        im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017
    else:
        im[0, :, :] = (im[0, :, :] - 103.94) * 0.017
        im[1, :, :] = (im[1, :, :] - 116.78) * 0.017
        im[2, :, :] = (im[2, :, :] - 123.68) * 0.017
    return im


def preprocess_image_for_imagenet_without_channel_reverse(img):
    """
     Same as the original except not reversing the channel.
    i.e. Only zero center by mean pixel.

    Parameters
    ----------
    img : ndarray with rank 3

    Returns
    -------
    img : ndarray with same shape
    """

    dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    x = img
    if dim_ordering == 'th':
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x


def preprocess_image_for_imagenet(img):
    """
     preprocessing_function: function that will be implied on each input.

            The function will run before any other modification on it.
            The function should take one argument: one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.

    Parameters
    ----------
    img : ndarray with rank 3

    Returns
    -------
    img : ndarray with same shape
    """

    dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    x = img
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x