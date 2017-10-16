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
import os
from datetime import datetime

from keras.models import model_from_json, load_model
from kyu.configs.engine_configs import RunningConfig
from kyu.configs.model_configs import get_model_config_by_name, MODEL_CONFIG_CLASS
from kyu.datasets import get_dataset_by_name
from kyu.engine.trainer import ClassificationTrainer
from kyu.experiment.general_train import get_dirhelper, get_argparser, get_data_generator

MODEL_FOLDER = '/cvlabdata1/home/kyu/so_updated_record/output/run/cls/ImageNet'
TESTING_DIRECTORY = 'MPN-RESNET50-baselineMPN-Cov-baseline no normalization_2017-09-07T15:14:33'


def recover_model_from_model_file(model_file):
    model = load_model(model_file, custom_objects=get_custom_objects())
    # decode the running epoch number
    filename = model_file.split('/')[-1]
    epoch = int(filename.split('.')[0].split('-')[0])
    return model, epoch


def recover_model_from_folder(model_folder):
    """

    check the model config by keras, load the model directly from it

    Parameters
    ----------
    model_folder

    Returns
    -------

    """
    if not os.path.exists(model_folder):
        raise IOError("File not found ! {}".format(model_folder))
    # Check the model file exists or not
    model_file_list = glob.glob(model_folder + '/keras_model.*.hdf5')
    if len(model_file_list) > 0:
        return recover_model_from_model_file(model_file_list[-1])

    # Load keras configs
    keras_config = glob.glob(model_folder + '/*keras.config')
    import json
    with open(keras_config[0], 'r') as f:
        f_json = json.load(f)
        json_config = json.loads(f_json)

    model = model_from_json(json_config, custom_objects=get_custom_objects())
    weight_path_list = glob.glob(model_folder + '/*.tmp')
    weight_path_list += glob.glob(model_folder + '/*weight*')
    # Choose one of it.
    if len(weight_path_list) > 1:
        print (weight_path_list)
    model.load_weights(weight_path_list[-1])
    Warning("Not finished yet for epoch")
    return model, 1


def recover_running_config(model_folder):
    run_config_path = glob.glob(model_folder + '/*run.config')
    return RunningConfig.load_config_from_file(run_config_path)


def recover_model_config(model_folder, model_config_cls):
    model_config_path = glob.glob(model_folder + '/*model.config')
    return model_config_cls.load_config_from_file(model_config_path)


def resume_train_with_data(
        dataset, model_class, model_config_class, runid,
        comments='', tf_dbg=False, tensorboard=False, **kwargs):

    # recover dataset
    data = get_dataset_by_name(str(dataset).lower())

    # Recover dir helper
    dirhelper = get_dirhelper(data.name, model_class)
    dirhelper.resume_runid(runid)

    rundir = dirhelper.get_model_run_path()

    # Write different cases with exceptions.
    model, initial_epoch = recover_model_from_folder(rundir)

    # Get running configs
    running_config = recover_running_config(rundir)
    running_config.comments += '\n {} {}'.format(datetime.now().isoformat().split('.')[0] ,comments)

    # Get model config
    model_config = recover_model_config(rundir, get_model_config_by_name(model_config_class))

    # Get data generator
    if running_config.train_image_gen_configs:
        data.train_image_gen_configs = running_config.train_image_gen_configs
    if running_config.valid_image_gen_configs:
        data.valid_image_gen_configs = running_config.valid_image_gen_configs

    train_image_gen = get_data_generator(model_config, data, mode='train')
    valid_image_gen = get_data_generator(model_config, data, mode='valid')
    dirhelper.build(running_config.title + model_config.name)

    trainer = ClassificationTrainer(model, data, dirhelper,
                                    model_config=model_config, running_config=running_config,
                                    save_log=True,
                                    logfile=dirhelper.get_log_path(),
                                    train_image_gen=train_image_gen,
                                    valid_image_gen=valid_image_gen)
    print("Succesfully resumed from the training model \n "
          "with running config {} \n"
          "model_config {} \n"
          ""
          )
    # trainer.model.summary()
    trainer.fit(verbose=running_config.verbose, initial_epoch=initial_epoch)
    trainer.plot_result()


if __name__ == '__main__':
    # Argument parser definition.
    parser = get_argparser()
    parser.add_argument('--runid', required=True, type=str)
    parser.add_argument('--model_config_class', '-mcc', required=True, type=str, choices=MODEL_CONFIG_CLASS.keys())
    args = parser.parse_args()

    resume_train_with_data(**vars(args))
