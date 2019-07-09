from .. import speaker_identification

import voxceleb1


class SubjectDataProducer(speaker_identification.DataProducer):

    def __init__(self, num_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

    def sample_subjects(self):
        """Return training and testing sets for metric learning.

        Output :
        data_subj : torch Tensor of shape (P, K, 1, W, H), 
        """
        data, test = self.produce()
        data_subj = self._sample_subjects(*data)
        test_subj = self._sample_subjects(*test)
        return data_subj, test_subj

    # === PRIVATE ===

    def _sample_subject(self, X, Y):
        it = self._iter_sample_subjects(X, Y)
        return torch.stack(list(it), dim=0)

    def _iter_sample_subjects(self, X, Y):
        for uid in set(Y.tolist()):
            X_select = X[Y == uid]
            X_index = voxceleb1.utils.tensor_tools.rand_indices(len(X_select))
            yield X_select[X_index[:self.num_samples]]
        
        
