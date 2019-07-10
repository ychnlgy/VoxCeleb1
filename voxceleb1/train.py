from . import utils
from . import preprocess
from . import training


def train(
    speaker_id_config_path,
    speaker_dist_config_path,
    stat_path,
    data_path,
    log_path,
    cores,
    max_chunks,
    email
):
    if email is None:
        log = utils.Logger(log_path)
    else:
        log = utils.EmailLogger(email, log_path)
    with log:
        max_chunks_str = str(max_chunks) if max_chunks is not None else "all"
        log.write("Loading %s chunks of samples from\n<%s>" % (
            max_chunks_str, data_path
        ))
        cfile = utils.ChunkFile(data_path)
        samples = list(map(preprocess.Sample.from_list, cfile.load(max_chunks)))
        log.write("Loaded %d samples" % len(samples))
        training.train(
            speaker_id_config_path,
            speaker_dist_config_path,
            stat_path,
            cores,
            samples,
            log
        )
