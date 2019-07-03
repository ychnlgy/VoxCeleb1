import numpy
import tqdm


class FragmentFile:

    def __init__(self, fpath):
        self.fpath = fpath

    def save(self, arr):
        with open(self.fpath, "wb") as f:
            items = self.itemize(arr)
            numpy.save(f, len(items))
            for item in tqdm.tqdm(items, desc="Saving to %s" % self.fpath):
                numpy.save(f, numpy.array(item, dtype=object))

    def load(self, max_n=None):
        with open(self.fpath, "rb") as f:
            total = numpy.load(f).item()
            n = total if max_n is None else min(total, max_n)
            items = range(n)
            for _ in tqdm.tqdm(items, desc="Loading from %s" % self.fpath):
                for obj in self.process(numpy.load(f)):
                    yield obj

    # === PROTECTED ===

    def itemize(self, arr):
        return arr

    def process(self, obj):
        yield obj
