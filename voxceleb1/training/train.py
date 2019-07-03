"""Main pipeline for training."""


from . import speaker_identification
from . import speaker_distance
from .config import Config

import voxceleb1

def train(speaker_id_config_path, samples, log_path):
    with voxceleb1.utils.Logger(log_path) as log:
        speaker_id_config = Config(
            speaker_id_config_path
        )
        log.write(str(speaker_id_config))

        dataset = speaker_identification.Dataset.create(samples)
        speaker_id_producer = DataProducer(speaker_id_config.slice_size, dataset)
        model = speaker_identification.search_model(
            speaker_id_config.model,
            dataset.features,
            speaker_id_config.latent_size,
            speaker_id_producer.len_unique_labels()
        )
        log.write("Architecture:\n%s" % str(model))

        speaker_identification.train(speaker_id_config, speaker_id_producer, model, log)
        speaker_distance.train(None, dataset, model, log)
