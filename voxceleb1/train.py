from . import utils
from . import preprocess
from . import training


def train(speaker_id_config_path, data_path, log_path, max_chunks):
    with utils.Logger(log_path) as log:
        max_chunks_str = str(max_chunks) if max_chunks is not None else "all"
        log.write("Loading %s chunks of samples from\n<%s>" % (
            max_chunks_str, data_path
        ))
        cfile = utils.ChunkFile(data_path)
        samples = list(map(preprocess.Sample.from_list, cfile.load(max_chunks)))
        log.write("Loaded %d samples" % len(samples))
        training.train(speaker_id_config_path, samples, log)
