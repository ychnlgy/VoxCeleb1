"""Main pipeline for training."""


from . import speaker_identification
from . import speaker_distance
from .search_model import search_model
from .dataset import Dataset
from ..config import Config

import voxceleb1

def train(speaker_id_config_path, samples, log_path):
    log = voxceleb1.utils.Logger(log_path)
    
    speaker_id_config = Config(speaker_id_config_path)
    log.write(str(speaker_id_config))

    dataset = Dataset.create(samples)
    speaker_id_producer = DataProducer(speaker_id_config.slice_size, dataset)
    model = search_model(
        speaker_id_config.model,
        dataset.features,
        speaker_id_producer.len_unique_labels()
    )
    log.write(str(model))

    speaker_identification.train(config, speaker_id_producer, model, log)
    speaker_distance.train(config, dataset, model, log)
