"""
Define the file structure in this folder
"""
import gzip

import os
from datetime import datetime
import cPickle

def getProjectPath():
    """
    Get project path based on current evaluation.
    Returns
    -------

    """
    p = os.path.abspath(__file__)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    # print(p)
    return p


class ProjectFile(object):
    """
    ProjectFile is a file management class, which provides the structure of organized outputs, sources and other
    file location related operations

    It is an upgrade version of previous io_utils get xxx folder etc.

    The structure is looking like this.

    root-path/
        output/
            run/
                dataset/
                    run_name/ # Different with model name, could be comparison between
                        model mode.timestamp/
                            file_of_config.config
                            run-plots
                            model_nb_epoch.weights
                            history
            plot/
                dataset/
                    run_plot_type/
                        model_mode_time_stamp.plots

        models/
            soft_link_to_models

        historys/
            # copy of all history
            dataset/
                model_mode_time_stamp.history
    """
    def __init__(self, root_path='/home/kyu/secondstat', source_path='/home/kyu/.pycharm/keras'):
        # Locate the root structure
        self.root_path = root_path
        if not os.path.exists(source_path):
            self.source_path = None
        else:
            self.source_path = source_path

        if not os.path.exists(root_path):
            os.mkdir(root_path)

        self.output_path = os.path.join(self.root_path, 'output')
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def get_run_path(self, dataset, model, mode):
        """
        Get run path to save all
            run/dataset/model/mode + time_stamp/

        Parameters
        ----------
        dataset : str   dataset name
        model : str     model name
        mode : str      settings.

        Returns
        -------
        root_path/run_path
        """
        timestamp = datetime.now().isoformat()
        run_path = os.path.join('run', dataset, model, mode + '.' + timestamp)
        return os.path.join(self.output_path, run_path)


def save_array(array, name):
    import tables
    f = tables.open_file(name, 'w')
    atom = tables.Atom.from_dtype(array.dtype)
    ds = f.create_carray(f.root, 'data', atom, array.shape)
    ds[:] = array
    f.close()


def load_array(name):
    import tables
    f = tables.open_file(name)
    array = f.root.data
    a = np.empty(shape=array.shape, dtype=array.dtype)
    a[:] = array[:]
    f.close()
    return a


def cpickle_load(filename):
    """
    Support loading from files directly.
    Proved to be same speed, but large overhead.
    Switch to load generator.
    :param filename:
    :return:
    """
    if os.path.exists(filename):
        if filename.endswith(".gz"):
            f = gzip.open(filename, 'rb')
        else:
            f = open(filename, 'rb')
        # f = open(path, 'rb')
        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding="bytes")
        f.close()
        return data  # (data
    else:
        print("File not found under location {}".format(filename))
        return None


def cpickle_save(data, output_file, ftype='gz'):
    """
    Save the self.label, self.image before normalization, to Pickle file
    To the specific output directory.
    :param output_file:
    :param ftype
    :return:
    """

    if ftype is 'gz' or ftype is 'gzip':
        print("compress with gz")
        output_file = output_file + "." + ftype
        f = gzip.open(output_file, 'wb')
    elif ftype is '':
        f = open(output_file, 'wb')
    else:
        raise ValueError("Only support type as gz or '' ")
    cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return output_file


def get_project_dir():
    """
    Get the current project directory
    :return:
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
    # current_dir = os.path.dirname(current_dir)
    # current_dir = os.path.dirname(current_dir)
    return current_dir


def get_dataset_dir():
    """
    Get current dataset directory
    :return:
    """
    # current_dir = get_project_dir()
    # return os.path.join(current_dir, 'data')
    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    return os.path.join(datadir_base, 'datasets')


def get_absolute_dir_project(filepath):
    """
    Get the absolute dir path under project folder.
    :param dir_name: given dir name
    :return: the absolute path.
    """
    path = os.path.join(get_project_dir(), filepath)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return path


def get_weight_path(filename, dir='project'):
    if dir is 'project':
        path = get_absolute_dir_project('model_saved')
        return os.path.join(path, filename)
    elif dir is 'dataset':
        dir_base = os.path.expanduser(os.path.join('~', '.keras'))
        dir = os.path.join(dir_base, 'models')
        if not os.path.exists(dir):
            os.mkdir(dir)
        return os.path.join(dir, filename)


def get_plot_path(filename, dir='project'):
    if dir is 'project':
        path = get_absolute_dir_project('model_saved/plots')
        if not os.path.exists(path):
            os.mkdir(path)
    elif dir is 'dataset':
        dir_base = os.path.expanduser(os.path.join('~', '.keras'))
        path = os.path.join(dir_base, 'plots')
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        raise ValueError("Only support project and dataset as dir input")
    if filename == '' or filename is None:
        return path
    return os.path.join(path, filename)


def get_plot_path_with_subdir(filename, subdir, dir='project'):
    """
    Allow user to get plot path with subdir

    Parameters
    ----------
    filename
    subdir : str    support
        'summary' for summary graph, 'run' for run-time graph, 'models' for model graph
    dir : str       support 'project' under project plot path, 'dataset' under ~/.keras/plots

    Returns
    -------
    path : str      absolute path of the given filename, or directory if filename is None or ''
    """
    path = get_plot_path('', dir=dir)
    path = os.path.join(path, subdir)
    if not os.path.exists(path):
        os.mkdir(path)
    if filename == '' or filename is None:
        return path
    return os.path.join(path, filename)


if __name__ == '__main__':
    getProjectPath()
    f_manage = ProjectFile(getProjectPath(), getProjectPath())
    p = f_manage.get_run_path('CIFAR', 'FitNet-v2_second', '10')
    print(p)
