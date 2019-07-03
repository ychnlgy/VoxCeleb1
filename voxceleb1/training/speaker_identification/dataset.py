class Dataset:

    @staticmethod
    def create(samples):
        assert len(samples) > 0
        data = [sample for sample in samples if sample.dev]
        test = [sample for sample in samples if not sample.dev]
        features = samples[0].spec.shape[0]  # shape (freq, time)
        return Dataset(data, test, features)

    def __init__(self, data, test, features):
        self.data = data
        self.test = test
        self.features = features
