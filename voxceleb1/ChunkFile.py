from .FragmentFile import FragmentFile


class ChunkFile(FragmentFile):

    def __init__(self, fpath, chunk_size):
        super().__init__(fpath)
        self.chunk_size = chunk_size

    # === PROTECTED ===

    def itemize(self, arr):
        arr = super().itemize(arr)
        chunks = range(0, len(arr), self.chunk_size)
        return [arr[i:i+self.chunk_size] for i in chunks]

    def process(self, chunk):
        for item in chunk:
            yield item
