import numpy
import torch

from .remapper import ReMapper


class DataProducer:

    def __init__(self, slice_size, dataset, random):
        self.remapper = ReMapper()
        self.dt = slice_size
        self.dataset = dataset
        self.random = random
        self._setup_remapper()

    def len_unique_labels(self):
        return len(self.remapper)
    
    def produce(self):
        """Constructs training and testing sets from the input Dataset.

        Note :
        This method should be run for each epoch of training,
        since it randomly slices samples from each variably-lengthed
        sample.

        Output:
        (
            (torch FloatTensor X_data, torch LongTensor Y_data),
            (torch FloatTensor X_test, torch LongTensor Y_test)
        ) : training and test sets and corresponding labels.
            X_data is of shape (N, 1, f, t), where N is the total
            number of samples in the training set, t is the time
            length of the spectrogram (i.e. time) and H is the height 
        """
        data = self._rand_select_input(self.dataset.data)
        test = self._rand_select_input(self.dataset.test)
        return data, test

    # === PRIVATE ===

    def _setup_remapper(self):
        for sample in self.dataset.data:
            self.remapper[sample.uid]
        self.remapper.freeze()

    def _rand_select_input(self, samples):
        inputs, Y = self._collect_input(samples)
        slices = list(map(self._rand_sample, inputs))
        X = torch.from_numpy(numpy.array(slices)).float().unsqueeze(1)
        Y = torch.from_numpy(numpy.array(Y)).long()
        return X, Y

    def _rand_sample(self, arr):
        """Assume input arr is of shape (freq, time)."""
        length = arr.shape[-1]
        assert length >= self.dt
        start = self.random.randint(0, length-self.dt)
        end = start + self.dt
        return arr[:, start:end]

    def _collect_input(self, samples):
        X = [s.spec.astype(numpy.float32) for s in samples]
        Y = [self.remapper[s.uid] for s in samples]
        return X, Y
