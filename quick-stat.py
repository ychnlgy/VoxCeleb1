import argparse
import collections

import voxceleb1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--save_stats", required=True)
    args = parser.parse_args()

    counter = collections.Counter()
    stats = voxceleb1.utils.BatchStat(axis=-1)
    for sample in voxceleb1.load(args.file):
        counter[sample.uid] += 1
        if sample.dev:
            stats.update(sample.spec)

    min_person_set = min(counter.values())
    print("Minimum person-specific samples:", min_person_set)

    with open(args.save_stats, "rb") as f:
        miu, std = stats.peek()
        numpy.save(f, numpy.array([miu, std]))
    print("Saved mean/std to %s." % args.save_stats)
