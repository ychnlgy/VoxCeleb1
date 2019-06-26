import argparse

from voxceleb1 import preprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--save_chunk_size", type=int, required=True)
    args = parser.parse_args()

    preprocess(args.root, args.outfile, args.save_chunk_size)
