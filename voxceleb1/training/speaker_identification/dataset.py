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
        dt = spec.shape[-1]
        i = self._rand.randint(0, dt - self._size)
        j = i + self._size
        X = (spec[:, i:j] - self._miu) / self._std
        return torch.from_numpy(X).float().unsqueeze(0), uid

    def get_uid(self, idx):
        return int(self._data[idx].uid)
