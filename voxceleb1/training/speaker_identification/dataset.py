import numpy
import torch
import tqdm

import torch.utils.data

import voxceleb1


class Dataset(torch.utils.data.Dataset):

    def __init__(self, samples, slice_size, miu, std, random, dev):
        super().__init__()
        self._size = slice_size
        self._rand = random
        self._miu = miu
        self._std = std
        self._dev = bool(dev)
        self._data = self._filter_samples(samples)
        self.features = self._data[0].spec.shape[0]

    def _filter_samples(self, samples):
        return [s for s in samples if bool(s.dev) == self._dev]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data[idx]
        spec, uid = sample.spec, sample.uid
        spec = torch.from_numpy(spec).float()
        dt = spec.size(-1)
        i = self._rand.randint(0, dt - self._size)
        j = i + self._size
        X = (spec[:, i:j] - self._miu) / self._std
        return X, uid
        
##class Dataset:
##
##    @staticmethod
##    def create(samples, log):
##        assert len(samples) > 0
##        log.write("Separating training/testing sets")
##        data = [sample for sample in samples if sample.dev]
##        test = [sample for sample in samples if not sample.dev]
##        features = samples[0].spec.shape[0]  # shape (freq, time)
##        return Dataset(data, test, features, log)
##
##    def __init__(self, data, test, features, log, eps=1e-8):
##        self.data = data
##        self.test = test
##        self.features = features
##        self.log = log
##        self.eps = eps
##
##        #self._normalize()
##
##    def _normalize(self):
##        self.log.write("Computing mean/std for the spectrogram filters")
##        miu, std = self._compute_stats(self.data)
##        self.log.write("Normalizing training set")
##        self._apply_normal(self.data, miu, std)
##        self.log.write("Normalizing testing set")
##        self._apply_normal(self.test, miu, std)
##
##    def _compute_stats(self, samples):
##        stat = voxceleb1.utils.BatchStat(axis=-1)
##        for sample in tqdm.tqdm(samples, desc="Compute stats", ncols=80):
##            stat.update(sample.spec)
##        return stat.peek()
##
##    def _apply_normal(self, samples, miu, std):
##        for sample in tqdm.tqdm(samples, desc="Normalizing", ncols=80):
##            sample.spec = (sample.spec - miu) / std
