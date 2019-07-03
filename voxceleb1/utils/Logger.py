import sys
import time


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
        prefix = "[{}-{:0>2}-{:0>2}-{:0>2}:{:0>2}:{:0>2}] ".format(
            tm.tm_year,
            tm.tm_mon,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec
        )
        msg = prefix + msg + end
        self.fout.write(msg)
        self.pout.write(msg)
