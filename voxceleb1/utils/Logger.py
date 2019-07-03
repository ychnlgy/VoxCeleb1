import sys
import time

from .format_time import format_time

class Logger:

    def __init__(self, fpath, pout=sys.stderr):
        self.fpath = fpath
        self.pout = pout
        self.fout = None

    def __enter__(self):
        self.fout = open(self.fpath, "a")
        return self

    def __exit__(self, *args):
        self.fout.close()

    def write(self, msg, end="\n"):
        tm = time.localtime()
        prefix = "[%s] " % format_time()
        msg = prefix + msg + end
        self.fout.write(msg)
        self.pout.write(msg)
