from __future__ import absolute_import
from __future__ import print_function

import logging
import tarfile
import os
import sys
import shutil
import hashlib
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError, HTTPError

from ..utils.generic_utils import Progbar


# Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
# urllib module, known to have issues with proxy management
if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        def chunk_read(response, chunk_size=8192, reporthook=None):
            total_size = response.info().get('Content-Length').strip()
            total_size = int(total_size)
            count = 0
            while 1:
                chunk = response.read(chunk_size)
                count += 1
                if not chunk:
                    reporthook(count, total_size, total_size)
                    break
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                yield chunk

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def get_file_from_dir(fname, directory):
    datadir = directory
    fpath = os.path.join(datadir, fname)
    if not os.path.exists(fpath):
        return None
    return fpath


def get_file(fname, origin, untar=False,
             md5_hash=None, cache_subdir='datasets'):
    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    logging.debug(datadir_base)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)
    download = False
    if os.path.exists(fpath):
        # file found; verify integrity if a hash was provided
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
        return untar_fpath

    return fpath


def validate_file(fpath, md5_hash):
    hasher = hashlib.md5()
    with open(fpath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False


def get_project_dir():
    """
    Get the current project directory
    :return:
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
    current_dir = os.path.dirname(current_dir)
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
        # os.mkdir(dir)
        return None
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
