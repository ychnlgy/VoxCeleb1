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
        self.write("======== INIT ========")
        return self

    def __exit__(self, typ, val, tb):
        if typ is not None:
            self.write(str(tb))
            self.write("%s: exit code %s" % (typ.__name__, val))
        self.write("======== DONE ========")
        self.fout.close()

    def write(self, msg, end="\n"):
        tm = time.localtime()
        prefix = "[%s] " % format_time()
        msg = prefix + msg + end
        self.fout.write(msg)
        self.pout.write(msg)
