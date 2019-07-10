"""Main pipeline for training."""
import numpy

from . import speaker_identification
from . import speaker_distance
from .config import Config

from .. import utils

def train(
    speaker_id_config_path,
    speaker_dist_config_path,
    stat_path,
    cores,
    samples,
    log
):
    speaker_id_config = Config(
        speaker_id_config_path
    )
    log.write(str(speaker_id_config))

    with utils.SeedBubble(speaker_id_config.seed) as random:
        log.write(
            "Seeding random, numpy and torch with: " \
            "%d" % speaker_id_config.seed
        )

        miu, std = numpy.load(stat_path)
        miu = miu.reshape(-1, 1)
        std = std.reshape(-1, 1)

        log.write("Remapping subject IDs")
        remapper = speaker_identification.ReMapper()
        with remapper.activate(lock=False):
            for sample in samples:
                sample.uid = remapper[sample.uid]

        log.write("Instantiating datasets")
        dataset = speaker_identification.Dataset(
            samples,
            speaker_id_config.slice_size,
            miu, std,
            random,
            dev=True
        )
        log.write("Number of training samples: %d" % len(dataset))

        testset = speaker_identification.Dataset(
            samples,
            speaker_id_config.slice_size,
            miu, std,
            random,
            dev=False
        )
        log.write("Number of testing samples: %d" % len(dataset))

        log.write("Instantiating model")
        model = speaker_identification.search_model(
            speaker_id_config.model,
            dataset.features,
            speaker_id_config.latent_size,
            len(remapper)
        )
        log.write("Architecture:\n%s" % str(model))

        speaker_identification.train(
            speaker_id_config,
            dataset, testset,
            cores,
            model,
            log
        )

        model.cut_tail()  # feature extraction only

##        speaker_dist_config = Config(
##            speaker_dist_config_path
##        )
##        speaker_dist_producer = speaker_distance.SubjectDataProducer(
##            speaker_dist_config.num_samples,
##            dataset,
##            random
##        )
##        speaker_distance.train(
##            speaker_dist_config,
##            speaker_dist_producer,
##            model,
##            log
##        )
