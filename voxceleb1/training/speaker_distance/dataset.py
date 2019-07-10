import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, speaker_id_dataset, num_samples):
        self._dset = speaker_id_dataset
        self._num_samples = num_samples
        self._subject_map = []
        
    def __len__(self):
        return len(self._subject_map)
