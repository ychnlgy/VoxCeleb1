import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, speaker_id_dataset, num_samples, random):
        self._dset = speaker_id_dataset
        self._num_samples = num_samples
        self._subject_map = []
        self._rand = random
        
    def __len__(self):
        return len(self._subject_map)

    def __getitem__(self, i):
        subj_data = self._subject_map[i]
        self._rand.shuffle(subj_data)
        return self._collect(subj_data[:self._num_samples])

    def _map_subjects(self):
        pass

    def _collect(self, indices):
        specs = [self._dset[i][0] for i in indices]
        return torch.stack(specs, dim=0)
