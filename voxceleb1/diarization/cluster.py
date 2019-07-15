class Cluster:

    def __init__(self, index, slice_len, embedding, threshold):
        """Represents a single cluster.

        Parameters :
        index : int starting position of the slice.
        slice_len : int length of the slice.
        embedding : torch FloatTensor vector of size (latent_size),
            the speaker embedding for the entire audio sample.
        threshold : float threshold distance at which two speaker
            embeddings are deemed to refer to the same speaker.
        """
        self._start = index
        self._slice = slice_len
        self._embs = [embedding]
        self._threshold = threshold
        self._label = None
        self._average = embedding
        self._n = 1

    def average(self):
        return self._average

    def copy_label(self, cluster):
        self._label = cluster._label

    def get_label(self):
        return self._label

    def set_label(self, label):
        self._label = label

    def matches(self, embedding):
        dist = (self._average - embedding).norm().item()
        return dist < self._threshold

    def append(self, embedding, dt):
        self._embs.append(embedding)
        new_n = self._n + 1
        self._average = self._n/new_n * self._average + 1/new_n * embedding
        self._slice += dt

    def get_slice(self):
        return slice(self._start, self._start + self._slice)
        

    