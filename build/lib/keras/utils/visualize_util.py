import os
from .data_utils import get_plot_path, get_plot_path_with_subdir
try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import colors as colors
    from matplotlib import cm as cmx
except ImportError:
    raise RuntimeError("Cannot import matplotlib")


def plot_loss_acc(tr_loss, te_loss, tr_acc, te_acc, epochs=None, show=False,
                  names=('train', 'test'), xlabel='epoch', ylabel=('loss', 'mis-class'),
                  linestyle=('dotted', '-'), color=('b', 'g'),
                  filename='default.png'
                  ):
    assert len(tr_loss) == len(te_loss)
    assert len(tr_acc) == len(te_acc)
    if epochs is None:
        epochs = range(len(tr_loss))

    fig, ax1 = plt.subplots()
    # Plot the loss accordingly
    ax1.plot(epochs, tr_loss, color=color[0], label=names[0], linestyle=linestyle[0])
    ax1.plot(epochs, te_loss, color=color[0], label=names[1], linestyle=linestyle[1])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel=ylabel[0], color=color[0])
    for tl in ax1.get_yticklabels():
        tl.set_color(color[0])

    ax2 = ax1.twinx()
    ax2.plot(epochs, tr_acc, color=color[1], label=names[0], linestyle=linestyle[0])
    ax2.plot(epochs, te_acc, color=color[1], label=names[1], linestyle=linestyle[1])
    ax2.set_ylabel(ylabel=ylabel[1], color=color[1])
    for tl in ax2.get_yticklabels():
        tl.set_color(color[1])

    leg2 = ax2.legend(loc=1, shadow=True)
    leg2.draw_frame(False)
    leg1 = ax1.legend(loc=1, shadow=True)
    leg1.draw_frame(False)

    plt.title(filename)

    if show:
        plt.show()
    plt_path = get_plot_path_with_subdir("train_test " + filename, subdir='run')
    plt.savefig(plt_path)
    plt.close()
    return plt_path

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def plot_multiple_train_test(train_errors, test_errors, modelnames, x_factor=None, show=False,
                             xlabel='', ylabel='', filename='', linestyle=('dotted', '-'),
                             significant=None, sig_color=None,
                             xlim=[0, 200], ylim=[0.01, 0.5]):
    assert len(train_errors) == len(test_errors) == len(modelnames)
    if x_factor is None:
        x_factor = range(len(train_errors))
    cmap = get_cmap(len(train_errors) + 1)
    if significant is not None:
        assert len(significant) == len(sig_color)
        tr_err = []
        te_err = []
        model_n = []
        sig_tr = []
        sig_te = []
        sig_model = []
        for i in range(len(train_errors)):
            if i in significant:
                sig_tr.append(train_errors[i])
                sig_te.append(test_errors[i])
                sig_model.append(modelnames[i])
            else:
                tr_err.append(train_errors[i])
                te_err.append(test_errors[i])
                model_n.append(modelnames[i])
        cmap = get_cmap(len(tr_err))
    else:
        tr_err = train_errors
        te_err = test_errors

    for i in range(len(tr_err)):
        col = cmap(i)
        plt.plot(range(len(tr_err[i])), tr_err[i], color=col, linestyle=linestyle[0])
        plt.plot(range(len(te_err[i])), te_err[i], color=col, linestyle=linestyle[1], label=modelnames[i])
    if significant is not None:
        for i in range(len(sig_tr)):
            plt.plot(range(len(sig_tr[i])), sig_tr[i], color=sig_color[i], linestyle=linestyle[0],
                     linewidth=3)
            plt.plot(range(len(sig_te[i])), sig_te[i], color=sig_color[i], linestyle=linestyle[1],
                     label=sig_model[i], linewidth=3)
    axes = plt.gca()
    axes.set_ylim(ylim)
    axes.set_xlim(xlim)
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(filename)
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='xx-small')

    if show:
        plt.show()
    plt_path = get_plot_path_with_subdir("train_test " + filename, subdir='summary')
    plt.savefig(plt_path)
    plt.close()
    return plt_path


def plot_train_test(train_errors, test_errors, x_factor=None, show=False,
                    names=('train', 'test'), xlabel='', ylabel='', filename='',
                    color=('b', 'r'), linestyle='-',
                    plot_type=0):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot
    :param plot_type: 0 for smooth, 1 for scatter
    """

    if plot_type == 0:
        plt.plot(x_factor, train_errors, color=color[0], label=names[0], linestyle=linestyle)
        plt.plot(x_factor, test_errors, color=color[1], label=names[1], linestyle=linestyle)
    elif plot_type == 1:
        plt.semilogx(x_factor, train_errors, color='b', marker='*', linestyle=linestyle, label=names[0])
        plt.semilogx(x_factor, test_errors, color='r', marker='*', linestyle=linestyle, label=names[1])
    else:
        raise RuntimeError("Unidentified plot type, must be either smooth or scatter")

    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(filename)

    if show:
        plt.show()
    plt_path = get_plot_path_with_subdir("train_test " + filename, subdir='run')
    plt.savefig(plt_path)
    plt.close()
    return plt_path


def model_to_dot(model, show_shapes=False, show_layer_names=True):
    try:
        # pydot-ng is a fork of pydot that is better maintained
        import pydot_ng as pydot
    except ImportError:
        # fall back on pydot if necessary
        import pydot

    if hasattr(pydot, 'find_graphviz'):
        if not pydot.find_graphviz():
            raise RuntimeError('Failed to import pydot. You must install pydot'
                               ' and graphviz for `pydotprint` to work.')
    else:
        pydot.Dot.create(pydot.Dot())

    # if not pydot.find_graphviz():
    #     raise RuntimeError('Failed to import pydot. You must install pydot'
    #                        ' and graphviz for `pydotprint` to work.')

    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if model.__class__.__name__ == 'Sequential':
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # first, populate the nodes of the graph
    for layer in layers:
        layer_id = str(id(layer))
        if show_layer_names:
            label = str(layer.name) + ' (' + layer.__class__.__name__ + ')'
        else:
            label = layer.__class__.__name__

        if show_shapes:
            # Build the label that will actually contain a table with the
            # input/output
            try:
                outputlabels = str(layer.output_shape)
            except:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)

    # second, add the edges
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                # add edges
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot(model, to_file='model.png', show_shapes=False, show_layer_names=True):
    try:
        # pydot-ng is a fork of pydot that is better maintained
        import pydot_ng as pydot
    except ImportError:
        # fall back on pydot if necessary
        import pydot

    if hasattr(pydot, 'find_graphviz'):
        if not pydot.find_graphviz():
            raise RuntimeError('Failed to import pydot. You must install pydot'
                               ' and graphviz for `pydotprint` to work.')
    else:
        pydot.Dot.create(pydot.Dot())

    # if not pydot.find_graphviz():
    #     raise RuntimeError('Failed to import pydot. You must install pydot'
    #                        ' and graphviz for `pydotprint` to work.')

    dot = model_to_dot(model, show_shapes, show_layer_names)
    _, format = os.path.splitext(to_file)
    if not format:
        format = 'png'
    else:
        format = format[1:]
    dot.write(to_file, format=format)
