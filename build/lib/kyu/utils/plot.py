import os

from example_engine import ExampleEngine
# from kyu.theano.cifar.cifar10_sndstat import cifar_fitnet_v2


def plot_cifar_fitnet(folder_name, significant_model=None,
                      log_history=None, log_model_name=None,
                      xlim=(0,200), ylim=(0, 1)):

    from example_engine import gethistoryfiledir
    hist_dir = gethistoryfiledir()
    hist_dir = os.path.join(hist_dir, folder_name)
    hist_list = []
    model_name_list = []

    # Load history from gz file
    for filename in os.listdir(hist_dir):
        if filename.endswith('.gz'):
            # file_list.append(filename)
            try:
                hist_dict = ExampleEngine.load_history(os.path.join(hist_dir, filename))
            except IOError:
                continue
            model_name = filename[filename.find('-') + 1: -filename.split('_')[-1].__len__() - 1]
            model_name_list.append(model_name)
            hist_list.append(hist_dict)

    # Load history from log file
    if log_history is not None and log_model_name is not None:
        assert len(log_history) == len(log_model_name)
        for i, log_file in enumerate(log_history):
            try:
                hist_dict = ExampleEngine.load_history_from_log(os.path.join(hist_dir, log_file))
            except IOError:
                continue
            model_name_list.append(log_model_name[i])
            hist_list.append(hist_dict)

    # For each plot: decode into title-model-hash.history
    from keras.utils.visualize_util import plot_multiple_train_test
    list_tr_mis = []
    list_te_mis = []
    list_tr_loss = []
    list_te_loss = []
    sig_id = []

    # Prepare for plotting
    for ind, name in enumerate(model_name_list):
        hist_dict = hist_list[ind]
        tr_mis = [1.0 - acc for acc in hist_dict['acc']]
        te_mis = [1.0 - acc for acc in hist_dict['val_acc']]
        tr_loss = hist_dict['loss']
        te_loss = hist_dict['val_loss']
        list_tr_loss.append(tr_loss)
        list_te_loss.append(te_loss)
        list_tr_mis.append(tr_mis)
        list_te_mis.append(te_mis)
        if significant_model is not None:
            if name in significant_model:
                sig_id.append(ind)
        # if loss_acc:
        #   plot_loss_acc(hist_dict['loss'], hist_dict['val_loss'],
        #                            tr_mis, te_mis,
        #                            filename=name+'.png'
        #                            )

    plot_multiple_train_test(list_tr_mis, list_te_mis, model_name_list,
                             xlabel='epoch', ylabel='mis-classification',
                             filename=folder_name,
                             significant=sig_id, sig_color=('k', 'b', 'r'),
                             # xlim=[0,200], ylim=[0, 1]
                             xlim=xlim, ylim=ylim
                             )


def plot_rescov_results():
    from example_engine import gethistoryfiledir
    hist_dir = gethistoryfiledir()
    # folder_name = 'Fitnet_CIFAR'
    folder_name = 'ResCov_CIFAR'
    hist_dir = os.path.join(hist_dir, folder_name)
    file_list = []
    model_list = []

    for filename in os.listdir(hist_dir):
        if filename.endswith('.gz'):
            file_list.append(filename)
            # later = str.join(filename.split('-')[1:])
            # model_name = str.join(later.split('_')[:-1])
            # ind1 = filename.find('-')
            # ind2 = filename.find('_', -filename.split('_')[-1].__len__(), -1)
            model_name = filename[filename.find('-') + 1: -filename.split('_')[-1].__len__() - 1]
            model_list.append(model_name)

    # For each plot: decode into title-model-hash.history
    from keras.utils.visualize_util import plot_multiple_train_test
    list_tr_mis = []
    list_te_mis = []
    list_tr_loss = []
    list_te_loss = []
    valid_model = []
    for ind, name in enumerate(model_list):
        try:
            hist_dict = ExampleEngine.load_history(os.path.join(hist_dir, file_list[ind]))
        except IOError:
            continue
        # print(hist_dict)
        tr_mis = [1.0 - acc for acc in hist_dict['acc']]
        te_mis = [1.0 - acc for acc in hist_dict['val_acc']]
        tr_loss = hist_dict['loss']
        te_loss = hist_dict['val_loss']
        list_tr_loss.append(tr_loss)
        list_te_loss.append(te_loss)
        list_tr_mis.append(tr_mis)
        list_te_mis.append(te_mis)
        valid_model.append(name)
        # plot_loss_acc(hist_dict['loss'], hist_dict['val_loss'],
        #                        tr_mis, te_mis,
        #                        filename=name+'.png'
        #                        )
    log_history = ['cifar10-ResCov_CIFAR_para-mode_5_0000.log',
                   'cifar10-ResCov_CIFAR_para-100_50_mode_5_0.log',
                   'cifar10-resnet50-baseline.log']
    log_model = ['ResCov_CIFAR_para-mode_5',
                 'ResCov_CIFAR_para-100_50-mode5',
                 'resnet50-baseline']
    for ind, log_file in enumerate(log_history):
        hist_dict = ExampleEngine.load_history_from_log(
            os.path.join(hist_dir, log_file))
        tr_mis = [1.0 - acc for acc in hist_dict['acc']]
        te_mis = [1.0 - acc for acc in hist_dict['val_acc']]
        tr_loss = hist_dict['loss']
        te_loss = hist_dict['val_loss']
        valid_model.append(log_model[ind])
        list_tr_loss.append(tr_loss)
        list_te_loss.append(te_loss)
        list_tr_mis.append(tr_mis)
        list_te_mis.append(te_mis)

    # model_list += log_model
    sig_id = (len(list_tr_mis) - 2, len(list_tr_mis) - 1)
    plot_multiple_train_test(list_tr_mis, list_te_mis, valid_model,
                             xlabel='epoch', ylabel='mis-classification',
                             filename='models-cifar10-ResCov',
                             significant=sig_id, sig_color=('r', 'k')
                             )


def plot_models():
    """
    Routine to plot models in Keras
    Returns
    -------

    """
    # parametrics = [[], [50,], [100,],[100, 50]]
    parametrics = [[],]
    nb_classes = 10
    from keras.utils.visualize_util import plot
    from keras.utils.data_utils import get_plot_path_with_subdir
    for mode in range(0, 6):
        for para in parametrics:
            # model = ResCovNet50CIFAR(parametrics=para, nb_class=nb_classes, cov_mode=cov_mode)
            model = cifar_fitnet_v2(parametrics=para, mode=mode)
            path = get_plot_path_with_subdir(model.name + '.png', subdir='models', dir='project')
            plot(model, to_file=path)


if __name__ == '__main__':
    # log_history = ['cifar10-ResCov_CIFAR_para-mode_5_0000.log',
    #                'cifar10-ResCov_CIFAR_para-100_50_mode_5_0000.log',
    #                'cifar10-resnet50-baseline.log']
    #     log_model = ['ResCov_CIFAR_para-mode_5',
    #                  'ResCov_CIFAR_para-100_50-mode5'
    #                  'resnet50-baseline']


    # plot_rescov_results()
    plot_cifar_fitnet('Fitnet_v2_CIFAR10_non_para_cov-dense', significant_model=['cifar-fitnet_v2_baseline_0'])
    # plot_models()