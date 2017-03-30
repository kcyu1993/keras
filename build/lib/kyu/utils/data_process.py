"""
process the CIFAR dataset

"""


import glob
import os

import itertools
import numpy as np
from keras.utils.visualize_util import get_cmap, plot_cross_validation
from kyu.utils.example_engine import ExampleEngine

# ROOT_FOLDER = '/Users/kcyu/Dropbox/git/keras/model_saved/history/CIFAR10/CIFAR-10 retest'
# ROOT_FOLDER = '/Users/kcyu/Dropbox/git/keras/model_saved/history/CIFAR10/cv_o2t'
ROOT_FOLDER_cvo2t = '/Users/kcyu/Dropbox/git/keras/model_saved/history/CIFAR10/cv_o2t_new'
ROOT_FOLDER_beta = '/Users/kcyu/Dropbox/git/keras/model_saved/history/CIFAR10/cv_covbeta'
# ROOT_FOLDER_beta = '/Users/kcyu/Dropbox/git/keras/model_saved/history/CIFAR10/cv_covbeta_minc_singlebranch'
# ROOT_FOLDER_beta = '/Users/kcyu/Dropbox/git/keras/model_saved/history/CIFAR10/cv_covbeta_minc'
ROOT_FOLDER_alpha = '/Users/kcyu/Dropbox/git/keras/model_saved/history/CIFAR10/cv_covalpha_minc'
ROOT_FOLDER = '/Users/kcyu/Dropbox/git/keras/model_saved/history/CIFAR10/cv_o2t_multi'

plot1_labels=[
              'No O2T',
              'O2T(50)',
              # 'O2T(70)',
              'O2T(100)',
              # 'O2T(90)',
              'O2T(150)',
              ]


# keys2 = ['fitnet_v1_o2_para-64']
keys2 = ['fitnet_v1_o2_para-']

cv_keys_para = [
        'fitnet_v1_o2_para-mode_1',
        'fitnet_v1_o2_para-50_mode_1',
        # 'fitnet_v1_o2_para-70_mode_1',
        # 'fitnet_v1_o2_para-30_mode_1',
        # 'fitnet_v1_o2_para-90_mode_1',
        'fitnet_v1_o2_para-100_mode_1',
        'fitnet_v1_o2_para-150_mode_1',

        ]


cv_keys_beta = [
    'cv_covBeta_',
]

cv_keys_alpha = [
    'cv_covAlpha_'
]

def get_wv_from_file_name(filename, key1='o2transform_wv', key2='_pmean-fitnet', return_type='int'):
    loc_a = filename.find(key1)
    loc_a += key1.__len__()
    loc_b = filename.find(key2)
    if return_type == 'int':
        return int(filename[loc_a: loc_b])
    else:
        return filename[loc_a: loc_b]


def get_para_from_file_name(filename, key1='fitnet_v1_o2_para-', key2='_mode_1'):
    return get_wv_from_file_name(filename, key1, key2, return_type='')


def get_beta_from_file_name(filename, key1='_cv_covBeta_', key2='_o2transform_cov_'):
    return float(get_wv_from_file_name(filename, key1, key2, return_type=''))

def get_alpha_from_file_name(filename, key1='retrain_minc2500_cv_covAlpha_', key2='_o2transform_cov_'):
    return float(get_wv_from_file_name(filename, key1, key2, return_type=''))

def load_history_from_file_with_key(filename, key):
    hist = ExampleEngine.load_history(filename)
    return hist[key]


def load_files_from_folder(folder):
    f_list = glob.glob(folder + '/*.history.gz')
    return f_list


def plot_cov_beta_validation():
    folder_dir =ROOT_FOLDER_beta
    labels = 'beta'
    # labels = ['beta {}'.format(float(i) / 10) for i in range(1, 11, 1)]
    filename = 'FitNet Cov-Beta Cross Validation'
    # filename = 'Minc2500 Cov-Beta Single branch Cross Validation'
    # filename = 'Minc2500 Cov-Beta Cross Validation'
    exp_lists = []
    f_list = load_files_from_folder(folder_dir)

    # Collect the experiments
    for ind, key in enumerate(cv_keys_beta):
        exp_lists.append([])
        for l in f_list:
            if key in l:
                exp_lists[ind].append(l)

    # Collect the data
    exp_datas = []
    exp_wvs = []
    exp_error = []
    for ind, f_l in enumerate(exp_lists):
        exp_datas.append([])
        exp_wvs.append([])
        exp_error.append([])
        for f in f_l:
            tmp_data = (load_history_from_file_with_key(
                os.path.join(folder_dir, f), 'val_acc')[-10:-1])
            if np.max(tmp_data) < 0.2:
                continue
            exp_error[ind].append(np.std(tmp_data))
            exp_datas[ind].append(np.max(tmp_data))
            exp_wvs[ind].append(get_beta_from_file_name(f))

    plot_cross_validation(exp_wvs, exp_datas, labels, linestyle='-',
                          filename=filename, error=exp_error,
                          x_lim=(0,1),
                          # y_lim=(0.72, 0.78),
                          y_lim=(0.80, 0.87),
                          smooth=False)

    print(exp_wvs)
    print(exp_datas)

def plot_cov_alpha_validation():
    folder_dir =ROOT_FOLDER_alpha
    labels = 'alpha'
    # labels = ['beta {}'.format(float(i) / 10) for i in range(1, 11, 1)]
    filename = 'Minc2500 Cov-Alpha Cross Validation'
    exp_lists = []
    f_list = load_files_from_folder(folder_dir)

    # Collect the experiments
    for ind, key in enumerate(cv_keys_alpha):
        exp_lists.append([])
        for l in f_list:
            if key in l:
                exp_lists[ind].append(l)

    # Collect the data
    exp_datas = []
    exp_wvs = []
    exp_error = []
    for ind, f_l in enumerate(exp_lists):
        exp_datas.append([])
        exp_wvs.append([])
        exp_error.append([])
        for f in f_l:
            tmp_data = (load_history_from_file_with_key(
                os.path.join(folder_dir, f), 'val_acc')[-10:-1])
            if np.max(tmp_data) < 0.2:
                continue
            exp_error[ind].append(np.std(tmp_data))
            exp_datas[ind].append(np.max(tmp_data))
            exp_wvs[ind].append(get_alpha_from_file_name(f))

    plot_cross_validation(exp_wvs, exp_datas, labels, linestyle='-',
                          filename=filename, error=exp_error,
                          x_lim=(0,1),
                          y_lim=(0.72, 0.78),
                          smooth=False)

    print(exp_wvs)


def plot_wv_cross_validation():
    folder_dir = ROOT_FOLDER_cvo2t
    labels = plot1_labels
    filename='FitNet PV Cross Validation'
    plot_cross_validation_based_on_key(folder_dir, cv_keys_para, labels, filename)


def plot_cross_validation_based_on_key(folder_dir, keys, plot_labels, title):
    exp_lists = []
    f_list = load_files_from_folder(folder_dir)

    # Collect the experiments
    for ind, key in enumerate(keys):
        exp_lists.append([])
        for l in f_list:
            if key in l:
                exp_lists[ind].append(l)

    # Collect the data
    exp_datas = []
    exp_wvs = []
    exp_error = []
    for ind, f_l in enumerate(exp_lists):
        exp_datas.append([])
        exp_wvs.append([])
        exp_error.append([])
        for f in f_l:
            tmp_data = (load_history_from_file_with_key(
                os.path.join(folder_dir, f), 'val_acc')[-8:-1])
            if np.average(tmp_data) < 0.2:
                continue
            exp_error[ind].append(np.std(tmp_data))
            exp_datas[ind].append(np.average(tmp_data))
            exp_wvs[ind].append(get_wv_from_file_name(f))

    plot_cross_validation(exp_wvs, exp_datas, plot_labels, linestyle='-',
                          filename=title, error=exp_error,
                          smooth=False)

    print(exp_wvs)


def print_last_accuracy(n=8):
    exp_lists = []
    f_list = load_files_from_folder(ROOT_FOLDER)

    # Collect the experiments
    for ind, key in enumerate(keys2):
        exp_lists.append([])
        for l in f_list:
            if key in l:
                exp_lists[ind].append(l)

    # Collect the data
    exp_datas = []
    exp_paras = []
    exp_error = []
    for ind, f_l in enumerate(exp_lists):
        exp_datas.append([])
        exp_paras.append([])
        exp_error.append([])
        for f in f_l:
            tmp_data = (load_history_from_file_with_key(
                os.path.join(ROOT_FOLDER, f), 'val_acc')[-1 - n:-1])
            exp_error[ind].append(np.std(tmp_data))
            exp_datas[ind].append(np.average(tmp_data))
            exp_paras[ind].append(get_para_from_file_name(f))

    print(exp_paras)
    print(exp_datas)
    print(exp_error)

if __name__ == '__main__':
    # print_last_accuracy(8)
    # plot_wv_cross_validation()
    plot_cov_beta_validation()
    # plot_cov_alpha_validation()