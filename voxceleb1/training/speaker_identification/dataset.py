import numpy


class Dataset:

    @staticmethod
    def create(samples):
        assert len(samples) > 0
        data = [sample for sample in samples if sample.dev]
        test = [sample for sample in samples if not sample.dev]
        features = samples[0].spec.shape[0]  # shape (freq, time)
        return Dataset(data, test, features)

    def __init__(self, data, test, features, eps=1e-8):
        self.data = data
        self.test = test
        self.features = features
        self.eps = eps

        self._normalize()

    def _normalize(self):
        miu, std = self._compute_stats(self.data)
        self._apply_normal(self.data, miu, std)
        self._apply_normal(self.test, miu, std)

    def _compute_stats(self, samples):
        specs = [s.spec for s in samples]
        stacked = numpy.concatenate(specs, axis=-1)
        miu = stacked.mean(axis=-1).reshape(-1, 1)
        std = stacked.std(axis=-1).reshape(-1, 1)
        std[(std < self.eps) | ~numpy.isfinite(std)] = 1  # numerical stability 
        return miu, std

    def _apply_normal(self, samples, miu, std):
        for sample in samples:
            sample.spec = (sample.spec - miu) / std
