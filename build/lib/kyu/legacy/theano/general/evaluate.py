import keras.backend as K
from kyu.theano.general.finetune import create_summary_op_for_keras_model, get_tmp_weights_path
import numpy as np

def run_evaluate(fn_model, fn_finetune, input_shape, config,
                 image_gen=None, nb_classes=0,
                 nb_epoch_finetune=50, nb_epoch_after=50,
                 title='', verbose=(2,2),
                 monitor_measures=[],
                 monitor_classes=[],
                 optimizer=None,
                 lr_decay=True):
    """
    General routine for finetune running.

    Should deal with the temporal saving and loading process.

    Parameters
    ----------
    fn_model : function to get the model, with interface defined.
    fn_finetune
    input_shape
    config
    nb_epoch_finetune
    nb_epoch_after
    image_gen
    nb_classes
    title

    Returns
    -------

    """
    monitor = False
    if len(monitor_classes) > 0 and len(monitor_classes) > 0:
        monitor = True
    if optimizer is not None:
        opt1 = optimizer[0]
        opt2 = optimizer[1]
        lr1 = opt1.init_lr
        lr2 = opt2.init_lr
    else:
        opt1 = None
        opt2 = None
        lr1 = 0.01
        lr2 = 0.001
    kwargs = config.kwargs
    random_key = np.random.randint(1, 1000)



    print("Running experiment {}".format(config.exp))
    for param in config.params:
        for mode in config.mode_list:
            if len(config.cov_outputs) == 0:
                cov_output = param[-1]
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                K.clear_session()
                sess = K.get_session()
                title = config.title + \
                        '_cov_{}_wv{}_{}'.format(config.cov_branch, str(cov_output), config.cov_mode)
                if nb_epoch_finetune > 0:
                    with sess.as_default():
                        model = fn_model(parametrics=param, mode=mode, cov_branch=config.cov_branch,
                                         cov_mode=config.cov_mode,
                                         nb_classes=nb_classes, cov_branch_output=cov_output,
                                         input_shape=input_shape,
                                         last_avg=False,
                                         freeze_conv=True,
                                         cov_regularizer=config.cov_regularizer,
                                         nb_branch=config.nb_branch,
                                         last_conv_feature_maps=config.last_conv_feature_maps,
                                         epsilon=config.epsilon,
                                         vectorization=config.vectorization,
                                         **kwargs
                                         )
                        # Write config to file
                        config.to_configobj(folder=title, comments='finetune_{}-lr_{}-model_{}'.format(
                            title, lr1, model.name))
                        if monitor:
                            summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)

                        fn_finetune(model,
                                    title='finetune_' + title,
                                    nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune,
                                    batch_size=config.batch_size, early_stop=config.early_stop, verbose=verbose[0],
                                    image_gen=image_gen,
                                    optimizer=opt1,
                                    lr=lr1,
                                    lr_decay=lr_decay)
                        model.save_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))

                K.clear_session()
                sess2 = K.get_session()
                if nb_epoch_after > 0:
                    with sess2.as_default():
                        model = fn_model(parametrics=param, mode=mode, cov_branch=config.cov_branch,
                                         cov_mode=config.cov_mode,
                                         nb_classes=nb_classes, cov_branch_output=cov_output,
                                         input_shape=input_shape,
                                         last_avg=False,
                                         freeze_conv=False,
                                         cov_regularizer=config.cov_regularizer,
                                         nb_branch=config.nb_branch,
                                         last_conv_feature_maps=config.last_conv_feature_maps,
                                         epsilon=config.epsilon,
                                         vectorization=config.vectorization,
                                         **kwargs
                                         )
                        if nb_epoch_finetune > 0:
                            model.load_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))
                        config.to_configobj(folder=title, comments='retrain_{}-lr_{}-model_{}'.format(
                            title, lr1, model.name))
                        if monitor:
                            summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)
                        fn_finetune(model,
                                    title='retrain_' + title,
                                    nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after,
                                    batch_size=config.batch_size / 4, early_stop=config.early_stop, verbose=verbose[1],
                                    image_gen=image_gen,
                                    optimizer=opt2,
                                    lr_decay=lr_decay,
                                    lr=lr2)
                        # Write config to file
            for cov_output in config.cov_outputs:
                print("Run routine 1 param {}, mode {}, covariance output {}".format(param, mode, cov_output))
                K.clear_session()
                sess = K.get_session()
                title = config.title + \
                        '_cov_{}_wv{}_{}'.format(config.cov_branch, str(cov_output), config.cov_mode)
                if nb_epoch_finetune > 0:
                    with sess.as_default():
                        model = fn_model(parametrics=param, mode=mode, cov_branch=config.cov_branch,
                                         cov_mode=config.cov_mode,
                                         nb_classes=nb_classes, cov_branch_output=cov_output,
                                         input_shape=input_shape,
                                         last_avg=False,
                                         freeze_conv=True,
                                         cov_regularizer=config.cov_regularizer,
                                         nb_branch=config.nb_branch,
                                         last_conv_feature_maps=config.last_conv_feature_maps,
                                         epsilon=config.epsilon,
                                         vectorization=config.vectorization,
                                         **kwargs
                                         )
                        # Write config to file
                        config.to_configobj(folder=title, comments='finetune_{}-lr_{}-model_{}'.format(
                            title, lr1, model.name))
                        if monitor:
                            summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)

                        fn_finetune(model,
                                    title='finetune_' + title,
                                    nb_epoch_after=0, nb_epoch_finetune=nb_epoch_finetune,
                                    batch_size=config.batch_size, early_stop=config.early_stop, verbose=verbose[0],
                                    image_gen=image_gen,
                                    optimizer=opt1,
                                    lr=lr1,
                                    lr_decay=lr_decay)
                        model.save_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))

                K.clear_session()
                sess2 = K.get_session()
                if nb_epoch_after > 0:
                    with sess2.as_default():
                        model = fn_model(parametrics=param, mode=mode, cov_branch=config.cov_branch,
                                         cov_mode=config.cov_mode,
                                         nb_classes=nb_classes, cov_branch_output=cov_output,
                                         input_shape=input_shape,
                                         last_avg=False,
                                         freeze_conv=False,
                                         cov_regularizer=config.cov_regularizer,
                                         nb_branch=config.nb_branch,
                                         last_conv_feature_maps=config.last_conv_feature_maps,
                                         epsilon=config.epsilon,
                                         vectorization=config.vectorization,
                                         **kwargs
                                         )
                        if nb_epoch_finetune > 0:
                            model.load_weights(get_tmp_weights_path(model.name + '_' + str(random_key)))
                        config.to_configobj(folder=title, comments='retrain_{}-lr_{}-model_{}'.format(
                            title, lr1, model.name))
                        if monitor:
                            summary_op = create_summary_op_for_keras_model(model, monitor_classes, monitor_measures)
                        fn_finetune(model,
                                    title='retrain_' + title,
                                    nb_epoch_after=0, nb_epoch_finetune=nb_epoch_after,
                                    batch_size=config.batch_size/4, early_stop=config.early_stop, verbose=verbose[1],
                                    image_gen=image_gen,
                                    optimizer=opt2,
                                    lr_decay=lr_decay,
                                    lr=lr2)
                    # Write config to file
