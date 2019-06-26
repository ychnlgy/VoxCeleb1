from .ChunkFile import ChunkFile
from .Sample import Sample


def load(fpath):
    cfile = ChunkFile(fpath)
    return map(Sample.from_list, cfile.load())
