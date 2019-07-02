import sys

class Logger:

    def __init__(self, fpath, pout=sys.stderr):
        self.fpath = fpath
        self.pout = pout
        self.fout = open(self.fpath, "w")

    def write(self, msg, end="\n"):
        msg = msg + end
        self.fout.write(msg)
        self.pout.write(msg)
