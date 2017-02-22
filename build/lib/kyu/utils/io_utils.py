"""
Define the file structure in this folder
"""
import os
from datetime import datetime

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




if __name__ == '__main__':
    getProjectPath()
    f_manage = ProjectFile(getProjectPath(), getProjectPath())
    p = f_manage.get_run_path('CIFAR', 'FitNet-v2_second', '10')
    print(p)