# TODO Resume training pipeline
"""
Current design:
    Provide the folder
    load the weights and epochs
    resume the training by re-executing
    save the logs as usual.

    TODO
        Add a function for logging: append instead of overwrite
        same for history.

"""

import os
import glob

import argparse

from kyu.configs.engine_configs import RunningConfig

from keras.models import model_from_json
from kyu.experiment.general_inference import get_argparser
from kyu.models.secondstat import get_custom_objects

MODEL_FOLDER = '/cvlabdata1/home/kyu/so_updated_record/output/run/cls/ImageNet'
TESTING_DIRECTORY = 'MPN-RESNET50-baselineMPN-Cov-baseline no normalization_2017-09-07T15:14:33'


def recover_model_from_folder(model_folder):
    """

    check the model config by keras, load the model directly from it

    Parameters
    ----------
    model_folder

    Returns
    -------

    """
    # files = os.listdir(model_folder)
    keras_config = glob.glob(model_folder + '/*keras.config')
    # Load keras configs
    import json
    with open(keras_config[0], 'r') as f:
        json_config = json.loads(json.load(f))

    model = model_from_json(json_config, custom_objects=get_custom_objects())
    weight_path = glob.glob(model_folder + '/*.tmp')
    model.load_weights(weight_path)

    return model


def recover_running_config(model_folder):
    run_config_path = glob.glob(model_folder + '/*run.config')
    return RunningConfig.load_config_from_file(run_config_path)




def main(runid, **kwargs):

    # Write different cases with exceptions.
    model = recover_model_from_folder(rundir)

    # Get running configs
    running_config = recover_running_config(rundir)



if __name__ == '__main__':
    # Argument parser definition.
    parser = get_argparser()
    parser.add_argument('--runid', require=True, type=str)
