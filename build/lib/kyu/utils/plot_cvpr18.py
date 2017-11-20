"""
Plot for CVPR 2018 paper

"""
import itertools
try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import colors as colors
    from matplotlib import cm as cmx
except ImportError:
    raise RuntimeError("Cannot import matplotlib")
import numpy as np

from kyu.engine.trainer import load_history_from_log, load_history
from kyu.utils.dict_utils import merge_history
from kyu.utils.visualize_util import plot_multiple_train_test, plot_cross_validation, get_cmap

cvlab3 = '/Users/kcyu/mount/cvlabdata3/cvkyu/secondstat_final/output/run/cls/'
cvkyu = '/Users/kcyu/mount/cvkyu/secondstat_final/output/run/cls/'
cvkyu_up = '/Users/kcyu/mount/cvkyu/so_updated_record/output/run/cls/'


def matplotlib_font():
    return {'family': 'normal',
            'weight': 'bold',
            'size': '16'}


def comparison_bilinear_pv_convergence():

    bcnn_hist = cvkyu + 'DTD/vgg/B-CNN-VGGBCNN-Baseline-448/bilnear-448_2017-11-08T12:23:20/bilnear-448_2017-11-08T12:23:20.history.gz'
    bcnn_hist_ft = cvkyu + 'DTD/vgg/B-CNN-VGGBCNN-Baseline-448/bilnear-448_2017-11-08T16:56:29/bilnear-448_2017-11-08T16:56:29.history.tmp.gz'
    pv_hist_path = cvlab3 + 'DTD/vgg16/SO-VGG16_normWVBN-Cov-PV_final-BN-448-compare ' \
                             'pv-1_branch/pv-256_2017-11-05T14:14:51/pv-256_2017-11-05T14:14:51.log'
    mpn_path = '/Users/kcyu/mount/cvkyu/secondstat/output/run/cls/DTD/VGG16/MPN-Cov-baseline no normalization_2017-' \
               '08-22T16:12:10/MPN-Cov-baseline no normalization_2017-08-22T16:12:10.log'

    # load the history and merge
    bcnn_history = []
    bcnn_history.append(load_history(bcnn_hist))
    bcnn_history.append(load_history(bcnn_hist_ft))

    bcnn_hist = merge_history(*bcnn_history)
    pv_hist = load_history_from_log(pv_hist_path)
    mpn_hist = load_history_from_log(mpn_path)
    model_names = ['Bilinear Pooling', 'MPN-COV', 'SMSO Pooling']
    plt_path = plot_multiple_train_test(
        (bcnn_hist['loss'], mpn_hist['loss'], pv_hist['loss']), (bcnn_hist['val_loss'], mpn_hist['val_loss'], pv_hist['val_loss']),
        modelnames=model_names,
        xlim=[0, 50], title='Convergence comparison on DTD',
        ylim=[0.01, 4], filename='/Users/kcyu/mount/plots/converge-dtd.eps'
    )
    print(plt_path)
    pass


def comparison_minc_pv_convergence():
    hist_paths = (
        cvkyu_up + 'Minc2500/vgg16/B-CNN-VGG16BCNN-Baseline-lastBN_2017-09-27T09:26:45/' \
                                'B-CNN-VGG16BCNN-Baseline_2017-09-27T09:26:45.log',
        cvkyu_up + 'Minc2500/vgg16/MPN-VGG16MPN-Cov-baseline-lastBN_2017-09-27T09:22:08'
                   '/MPN-VGG16MPN-Cov-baseline no normalization_2017-09-27T09:22:08.log',
        cvlab3 + 'Minc2500/vgg16/SO-VGG16_normWVBN-Cov-PV2048_final-BN-1_branch/final-model-with-true-pv-no'
                 'rm,-nobias-and-no-gamma,-last-BN_2017-10-18T17:43:10/final-model-with-true-pv-norm,-nobia'
                 's-and-no-gamma,-last-BN_2017-10-18T17:43:10.log',
    )

    historys = []
    for p in hist_paths:
        historys.append(load_history_from_log(p))
    model_names = ['Bilinear Pooling', 'MPN-COV', 'SMSO Pooling']

    plt_path = plot_multiple_train_test(
        [hist['loss'] for hist in historys], [hist['val_loss'] for hist in historys],
        modelnames=model_names,
        xlim=[0, 50],
        title='Convergence comparison on MINC-2500',
        ylim=[0.01, 3], filename='/Users/kcyu/mount/plots/converge_minc_loss.eps',
    )
    print(plt_path)


def comparision_pv_convergence_dual():
    # Load MINC historys
    hist_paths = (
        cvkyu_up + 'Minc2500/vgg16/B-CNN-VGG16BCNN-Baseline-lastBN_2017-09-27T09:26:45/' \
                                'B-CNN-VGG16BCNN-Baseline_2017-09-27T09:26:45.log',
        cvkyu_up + 'Minc2500/vgg16/MPN-VGG16MPN-Cov-baseline-lastBN_2017-09-27T09:22:08'
                   '/MPN-VGG16MPN-Cov-baseline no normalization_2017-09-27T09:22:08.log',
        cvlab3 + 'Minc2500/vgg16/SO-VGG16_normWVBN-Cov-PV2048_final-BN-1_branch/final-model-with-true-pv-no'
                 'rm,-nobias-and-no-gamma,-last-BN_2017-10-18T17:43:10/final-model-with-true-pv-norm,-nobia'
                 's-and-no-gamma,-last-BN_2017-10-18T17:43:10.log',
    )

    minc_historys = []
    for p in hist_paths:
        minc_historys.append(load_history_from_log(p))

    # Load DTD historys
    bcnn_hist = cvkyu + 'DTD/vgg/B-CNN-VGGBCNN-Baseline-448/bilnear-448_2017-11-08T12:23:20/bilnear-448_2017-11-08T12:23:20.history.gz'
    bcnn_hist_ft = cvkyu + 'DTD/vgg/B-CNN-VGGBCNN-Baseline-448/bilnear-448_2017-11-08T16:56:29/bilnear-448_2017-11-08T16:56:29.history.tmp.gz'
    pv_hist_path = cvlab3 + 'DTD/vgg16/SO-VGG16_normWVBN-Cov-PV_final-BN-448-compare ' \
                            'pv-1_branch/pv-256_2017-11-05T14:14:51/pv-256_2017-11-05T14:14:51.log'
    mpn_path = '/Users/kcyu/mount/cvkyu/secondstat/output/run/cls/DTD/VGG16/MPN-Cov-baseline no normalization_2017-' \
               '08-22T16:12:10/MPN-Cov-baseline no normalization_2017-08-22T16:12:10.log'

    # load the history and merge
    bcnn_history = []
    bcnn_history.append(load_history(bcnn_hist))
    bcnn_history.append(load_history(bcnn_hist_ft))

    bcnn_hist = merge_history(*bcnn_history)
    pv_hist = load_history_from_log(pv_hist_path)
    mpn_hist = load_history_from_log(mpn_path)

    dtd_historys = []
    dtd_historys.append(bcnn_hist)
    dtd_historys.append(mpn_hist)
    dtd_historys.append(pv_hist)

    model_names = ['BP', 'MPN', 'SMSO Pooling']

    dtd_tr = [hist['loss'] for hist in dtd_historys]
    dtd_te = [hist['val_loss'] for hist in dtd_historys]
    minc_tr = [hist['loss'] for hist in minc_historys]
    minc_te = [hist['val_loss'] for hist in minc_historys]

    # Color map
    cmap = [[27, 158, 119],
            [41, 100, 229],
            [217, 95, 2],
            ]
    cmap = [[float(a) / 255 for a in b] for b in cmap]
    # line style
    linestyle = ['--', '-']

    # plot configs
    xlim = (0, 50)
    ylims = [(0.001, 4.2), (0.001, 3.2)]
    major_ticks = np.arange(0, 5, 1, dtype=np.float32)
    minor_ticks = np.arange(0.5, 4, 1, dtype=np.float32)
    xlabel = 'Num Epochs'
    ylabel = 'Loss'
    filename = '/Users/kcyu/mount/plots/convergence-all-v2.eps'

    # Define the subplot accordingly.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(10, 5))

    # ax 1 for dtd
    for i in range(len(dtd_tr)):
        col = cmap[i]
        ax1.plot(range(len(dtd_tr[i])), dtd_tr[i], color=col, linestyle=linestyle[0])
        ax1.plot(range(len(dtd_te[i])), dtd_te[i], color=col, linestyle=linestyle[1], label=model_names[i], lw=2)

    # ax 2 for minc
    for i in range(len(minc_tr)):
        col = cmap[i]
        ax2.plot(range(len(minc_tr[i])), minc_tr[i], color=col, linestyle=linestyle[0])
        ax2.plot(range(len(minc_te[i])), minc_te[i], color=col, linestyle=linestyle[1], label=model_names[i], lw=2)

    for ind, lplt in enumerate((ax1, ax2)):

        # lplt.xlabel(xlabel)
        # lplt.ylabel(ylabel)
        lplt.set_yticks(major_ticks)
        lplt.set_yticks(minor_ticks, minor=True)
        lplt.set_ylim(ylims[ind])
        lplt.set_xlim(xlim)
        lplt.grid(True, axis='y', ls='dotted', lw=.4, c='k', alpha=1)
        lplt.spines['top'].set_visible(False)
        lplt.spines['right'].set_visible(False)

        # set font size
        for item in ([lplt.title, lplt.xaxis.label, lplt.yaxis.label] +
                     lplt.get_xticklabels() + lplt.get_yticklabels()):
            item.set_fontsize(16)
    ax1.set_xlabel('(a)')
    ax2.set_xlabel('(b)')

    # Common y label
    fig.text(0.08, 0.5, 'Loss', va='center', rotation='vertical', fontsize=16)
    fig.text(0.5, 0.01, 'Num epochs', ha='center', fontsize=16)

    title = ''
    # plt.yscale('log')
    leg = plt.legend(loc=1, shadow=False)
    leg.draw_frame(True)
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=18)

    plt_path = filename
    print("save plot to " + plt_path)
    plt.savefig(plt_path)
    plt.close()
    return plt_path


def data_cross_pv():
    #       32      64      128         256     512     1024    2048        4k      8k
    DTD = [0.66586, 0.68187, 0.68981, 0.69006, 0.69635, 0.69067, 0.69206, 0.67891, 0.67348]
    MINC = [0.7492, 0.74187, 0.73557, 0.75096, 0.76154, 0.77012, 0.77999, 0.76503, 0.73012]
    MIT = [0.7228, 0.75378, 0.7613, 0.7694, 0.76946, 0.7878, 0.7945, 0.79381, 0.77417]

    DTD = [0.65504,] + DTD
    MINC = [0.7449,] + MINC
    MIT = [0.7114,] + MIT

    list_y = [DTD, MINC, MIT]
    list_x = [a for a in range(4, 14)]
    labels = ['DTD', 'MINC', 'MIT']
    list_x = [list_x, list_x, list_x]
    return list_x, list_y, labels


def cross_validate_pv():
    """
    First version cross validate PV.

    Returns
    -------

    """
    list_x, list_y, labels = data_cross_pv()
    # cmap = get_cmap(len(list_x) + 1)
    filename = '/Users/kcyu/mount/plots/pv_cross.eps'
    title = ''
    x_lim = (5,13)
    y_lim = (0.6, 0.90)
    xlabel = 'PV dimension log(p)'
    ylabel = 'top 1 accuracy'
    plt.rc('legend', fontsize=16)
    cmap = get_cmap(len(list_x) + 1)
    error = None
    showLegend = True
    show = True
    plotdot = ['-s', '-x', '-^']
    for ind, (d, wv) in enumerate(zip(list_y, list_x)):
        # sort the d and
        if error is not None:
            lists = sorted(itertools.izip(*[wv, d, error[ind]]))
            new_x, new_y, new_e = list(itertools.izip(*lists))
        else:
            lists = sorted(itertools.izip(*[wv, d]))
            new_x, new_y = list(itertools.izip(*lists))
        col = cmap(ind)
        if error is not None:
            new_e = np.asanyarray(new_e, np.float32)
            plt.errorbar(new_x, new_y, new_e, color=col, elinewidth=0.5, capsize=2, capthick=1)

        plt.plot(new_x, new_y, plotdot[ind],
                 color=col, linestyle='-', label=labels[ind],
                 ms=10, lw=2, alpha=0.7,
                 )

    axes = plt.gca()
    axes.set_ylim(y_lim)
    axes.set_xlim(x_lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if showLegend:
        leg = plt.legend(loc=1, shadow=False, fontsize=10, handleheight=2, handlelength=4)
        leg.draw_frame(True)
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize='medium')

    if show:
        plt.show()
    # update the file name system.
    plt_path = filename
    print("save plot to " + plt_path)
    plt.savefig(plt_path + '.eps', format='eps')
    plt.close()
    return plt_path


def cross_validate_pv_subplot():
    """
    Sub plot version of cross validation

    Returns
    -------
    plt-path
    """
    filename = '/Users/kcyu/mount/plots/pv_cross-sep-v3.eps'
    list_x, list_y, labels = data_cross_pv()

    data_label = labels
    labels = 3*['SMSO pooling']
    # Color map
    cmap = [[27, 158, 119],
            [117, 112, 179],
            [217, 95, 2],
            ]
    # cmap = 3*[[31,120,180],
    #         ]
    cmap = [[float(a) / 255 for a in b] for b in cmap]
    plotdot = ['-s', '-x', '-^']

    # configs for each sub-plot
    y_lims = [[0.65,0.72], [0.7,0.8], [0.71,0.86]]
    x_lim = [4, 14]
    # CBP-TS baseline
    baseline_x = 3*[range(0,20)]
    cbp_ts_baseline = [0.6771, 0.7240, 0.7683]
    baseline_y = [len(nb) * [cbp_ts_baseline[ind],] for ind, nb in enumerate(baseline_x)]

    # Create sub-plot
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(10,7))

    # Plot each sub-plot
    for ind, (d, wv) in enumerate(zip(list_y, list_x)):
        lists = sorted(itertools.izip(*[wv, d]))
        new_x, new_y = list(itertools.izip(*lists))
        col = cmap[ind]
        axes[ind].plot(
            new_x, new_y, plotdot[ind],
            color=col, linestyle='-', label=labels[ind],
            ms=10, lw=2, alpha=0.7,)
        # if ind == 0:
        axes[ind].plot(baseline_x[ind], baseline_y[ind], '--', label='CBP-TS', lw=1, alpha=0.3, color=col)
        # else:
        #     axes[ind].plot(baseline_x[ind], baseline_y[ind], '-.', lw=1, alpha=0.3, color='b')

        # axes[ind].grid(True, axis='y', ls='-', lw=.2, c='k', alpha=0.3)
        axes[ind].set_ylim(y_lims[ind])

        axes[ind].spines['top'].set_visible(False)
        axes[ind].spines['right'].set_visible(False)
        # legend
        leg = axes[ind].legend(loc=2, shadow=False, fontsize=14)
        leg.draw_frame(True)
        ltext = leg.get_texts()
        # axes[ind].setp(ltext, fontsize=18)

    plt.xlabel('PV dimension log(p)', fontsize=16)
    plt.xlim(x_lim)
    # Common y label
    fig.text(0.04, 0.5, 'Top 1 accuracy', va='center', rotation='vertical', fontsize=16)
    txt = 'VGG-D on '
    fig.text(0.50, 0.30, txt+ data_label[2], ha='center', fontsize=16)
    fig.text(0.50, 0.575, txt+data_label[1], ha='center', fontsize=16)
    fig.text(0.50, 0.84, txt+data_label[0], ha='center', fontsize=16)




    # legend
    # leg = plt.legend(shadow=False)
    # leg.draw_frame(True)
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=18)

    plt_path = filename
    print("save plot to " + plt_path)
    plt.savefig(plt_path)
    plt.close()
    return plt_path

if __name__ == '__main__':
    # comparison_bilinear_pv_convergence()
    # comparison_minc_pv_convergence()
    # cross_validate_pv()
    comparision_pv_convergence_dual()
    # cross_validate_pv_subplot()