from .ChunkFile import ChunkFile


def save(outfile, save_chunk_size, samples):
    cfile = ChunkFile(outfile, save_chunk_size)
    cfile.save([sample.to_list() for sample in samples])
