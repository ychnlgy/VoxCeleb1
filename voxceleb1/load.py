from . import utils
from . import preprocess


def load(fpath):
    cfile = utils.ChunkFile(fpath)
    return map(preprocess.Sample.from_list, cfile.load())
