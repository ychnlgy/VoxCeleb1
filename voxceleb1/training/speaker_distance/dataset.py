import collections

import torch
import torch.utils.data
import tqdm

class Dataset(torch.utils.data.Dataset):

    def __init__(self, speaker_id_dataset, num_samples, random):
        """Dataset for verifying speakers.

        Parameters :
        speaker_id_dataset : ../speaker_identification:Dataset instance.
        num_samples : number of samples per subject for random sampling.
        random : random:Random instance.
        """
        self._dset = speaker_id_dataset
        self._num_samples = num_samples
        self._subject_map = self._group_subjects()
        self._rand = random
        
    def __len__(self):
        return len(self._subject_map)

    def __getitem__(self, i):
        subj_data = self._subject_map[i]
        self._rand.shuffle(subj_data)
        return self._collect(subj_data[:self._num_samples])

    # === PRIVATE ===

    def _group_subjects(self):
        out = collections.defaultdict(list)
        for i in tqdm.tqdm(
            range(len(self._dset)),
            ncols=80,
            desc="Grouping by subject"
        ):
            out[self._dset.get_uid(i)].append(i)
        return list(out.values())

    def _collect(self, indices):
        specs = [self._dset[i][0] for i in indices]
        return torch.stack(specs, dim=0)
