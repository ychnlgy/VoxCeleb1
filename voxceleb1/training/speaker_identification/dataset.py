import numpy

import voxceleb1


class Dataset:

    @staticmethod
    def create(samples, log):
        assert len(samples) > 0
        log.write("Separating training/testing sets")
        data = [sample for sample in samples if sample.dev]
        test = [sample for sample in samples if not sample.dev]
        features = samples[0].spec.shape[0]  # shape (freq, time)
        return Dataset(data, test, features, log)

    def __init__(self, data, test, features, log, eps=1e-8):
        self.data = data
        self.test = test
        self.features = features
        self.log = log
        self.eps = eps

        self._normalize()

    def _normalize(self):
        self.log.write("Computing mean/std for the spectrogram filters")
        miu, std = self._compute_stats(self.data)
        self.log.write(
            "Applying normalization to the training and testing sets"
        )
        self._apply_normal(self.data, miu, std)
        self._apply_normal(self.test, miu, std)

    def _compute_stats(self, samples):
        stat = voxceleb1.utils.BatchStat(axis=-1)
        for sample in samples:
            stat.update(sample.spec)
        return stat.peek()

    def _apply_normal(self, samples, miu, std):
        for sample in samples:
            sample.spec = (sample.spec - miu) / std
