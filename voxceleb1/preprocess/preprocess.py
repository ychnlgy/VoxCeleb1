import tqdm

from .collect import collect
from .save import save


def preprocess(root, outfile, save_chunk_size):
    samples = collect(root)

    for sample in tqdm.tqdm(samples, "Loading raw audio"):
        sample.load()

    for sample in tqdm.tqdm(samples, "Creating spectrograms"):
        sample.transform()

    save(outfile, save_chunk_size, samples)
