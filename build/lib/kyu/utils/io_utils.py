"""
Define the file structure in this folder
"""
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
                    model/
                        mode.timestamp/
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




if __name__ == '__main__':
    getProjectPath()
    f_manage = ProjectFile(getProjectPath(), getProjectPath())
    p = f_manage.get_run_path('CIFAR', 'FitNet-v2_second', '10')
    print(p)