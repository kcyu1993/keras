import sys

from .io_utils import get_absolute_dir_project


class Logger(object):

    def __init__(self, filename=get_absolute_dir_project("model_saved/log/Default.log")):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.filename = filename

    def write(self, message):
        self.terminal.write(message)
        if not self.log.closed:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()

    def close(self):
        self.log.close()
        return self.terminal

    def open(self):
        self.log = open(self.filename, "a")
        return self
