"""Main pipeline for training."""


from . import speaker_identification
from . import speaker_distance
from .search_model import search_model

import voxceleb1


class TrainingParameters:

    def __init__(
        self,
        model,
        slice_size,
        speaker_id_epochs,
    ):
        self.model = model
        self.slice_size = slice_size
        self.speaker_id_epochs = speaker_id_epochs


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


def train(param_dict, samples):
    params = TrainingParameters(**param_dict)
    log = voxceleb1.utils.Logger(params.logpath)
    # Log parameters
    dataset = Dataset.create(samples)
    model = search_model(params.model, dataset.features)

    speaker_identification.train(params, dataset, model, log)
    speaker_distance.train(params, dataset, model, log)
