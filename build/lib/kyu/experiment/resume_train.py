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

import glob

from datetime import datetime

from keras.models import model_from_json
from kyu.configs.engine_configs import RunningConfig
from kyu.configs.model_configs import get_model_config_by_name, MODEL_CONFIG_CLASS
from kyu.datasets import get_dataset_by_name
from kyu.experiment.general_inference import get_argparser
from kyu.experiment.general_train import get_dirhelper
from kyu.layers.secondstat import get_custom_objects

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


def recover_model_config(model_folder, model_config_cls):
    model_config_path = glob.glob(model_folder + '/*model.config')
    return model_config_cls.load_config_from_file(model_config_path)


def resume_train_with_data(
        dataset, model_class, model_config_class, rundir,
        comments='', tf_dbg=False, tensorboard=False, **kwargs):

    dirhelper = get_dirhelper(dataset, model_class, **kwargs)
    dirhelper.resume_runid(rundir)

    rundir = dirhelper.get_model_run_path()

    # Write different cases with exceptions.
    model = recover_model_from_folder(rundir)

    # Get running configs
    running_config = recover_running_config(rundir)
    running_config.comments += '\n {} {}'.format(datetime.now().isoformat().split('.')[0] ,comments)

    # Get dataset
    dataset = get_dataset_by_name(dataset)

    # Get model config
    model_config = recover_model_config(rundir, get_model_config_by_name(model_config_class))



if __name__ == '__main__':
    # Argument parser definition.
    parser = get_argparser()
    parser.add_argument('--runid', require=True, type=str)
    parser.add_argument('--model_config_class', '-mcc', require=True, type=str, choices=MODEL_CONFIG_CLASS.keys())
