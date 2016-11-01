"""
Logger from Stack-overflow.
"""
from __future__ import absolute_import

import sys
from .data_utils import get_absolute_dir_project

class Logger(object):
    def __init__(self, filename=get_absolute_dir_project("model_saved/log/Default.log")):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

