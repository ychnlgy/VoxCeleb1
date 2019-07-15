import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, spec, slice_size, step_size):
        assert len(spec.shape) == 2
        self._spec = spec.unsqueeze(0)  # (1, freq, time)
        self._slice_size = slice_size
        self._step_size = step_size
        self._dt = spec.size(1)
        self._len = (self._dt - self._slice_size) // self._step_size

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        i = idx * self._step_size
        j = i + self._slice_size
        return self._spec[:, :, i:j]  # (1, freq, time)
