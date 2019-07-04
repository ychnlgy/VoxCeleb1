import numpy

from . import utils
from . import preprocess
from . import training

FREQS = 256

def train_toy(speaker_id_config_path, log_path, num_samples):
    with utils.Logger(log_path) as log:
        log.write("Creating %d samples" % num_samples)
        samples = list(iter_toy_samples(num_samples))
        training.train(speaker_id_config_path, samples, log)

def iter_toy_samples(num_samples):
    with utils.SeedBubble(seed=5) as random:
        total_subjects = 100

        for i in range(num_samples):
            uid = random.randint(1, total_subjects)
            yield preprocess.Sample.from_list(
                [
                    random.randint(0, 1),
                    uid,
                    None,
                    create_toy_spec(uid, random)
                ]
            )

def create_toy_spec(uid, random):
    length = random.randint(330, 1500)
    buf = numpy.zeros((FREQS, length))
    buf[0, 0] = uid
    return buf
