"""
Tensorflow Slim training pipeline for ImageNet model

TODO:
    Potentially to add Cov-layer implemented in keras to do further work.
"""

import tensorflow.contrib.slim as slim


# Download ImageNet data according to TFRecord format by Inception model.
# ImageNet username and access key
#   Username = kcyu2015
#   Access K = 1dd6fb0e88bc061af710efe16dc523bf43ba8058

"""
ImageNet Data-building

1. Use TensorFlow models/inception to build TFRecord for ImageNet
    a. Download the original dataset, unzip to specific location
    b. Unify structure, make validation and train the same
        train or valid/
        |-- wnid/
        |   |-- imgs
    c. Build ImageNet with models.inception.data.build_imagenet

2.
"""

