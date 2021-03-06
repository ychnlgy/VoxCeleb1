import sys
import time
import traceback

from .format_time import format_time

class Logger:

    def __init__(self, fpath, pout=sys.stderr):
        self.fpath = fpath
        self.pout = pout
        self.fout = None

    def __enter__(self):
        self.fout = open(self.fpath, "a")
        self.write("======== START ========")
        return self

    def __exit__(self, typ, val, tb):
        state = "success"
        if typ is not None:
            self.write("ERROR:\n%s" % traceback.format_exc(), silent=True)
            state = "crashed"
        self.write("======== EXIT: %s ========" % state)
        self.fout.close()

    def write(self, msg, end="\n", silent=False):
        tm = time.localtime()
        prefix = "[%s] " % format_time()
        msg = prefix + msg + end
        self.fout.write(msg)
        self.fout.flush()
        if not silent:
            self.pout.write(msg)
