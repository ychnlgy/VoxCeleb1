from . import utils
from . import preprocess
from . import training


def train(speaker_id_config_path, data_path, log_path, max_chunks):
    with utils.Logger(log_path) as log:
        log.write("Loading %d chunks of samples from\n%s" % (
            max_chunks, data_path
        ))
        cfile = utils.ChunkFile(data_path)
        samples = list(map(preprocess.Sample.from_list, cfile.load(max_chunks)))
        training.train(speaker_id_config_path, samples, log)
