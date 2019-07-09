import argparse
import collections

from voxceleb1 import load

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    counter = collections.Counter()
    for sample in load(args.file):
        counter[sample.uid] += 1
    min_person_set = min(counter.values())
    print("Minimum person-specific samples:", min_person_set)
