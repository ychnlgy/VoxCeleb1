import argparse
import collections
import statistics

from voxceleb1 import load

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    samples = list(load(args.file))

    print("Number of samples:", len(samples))
    print("Features:", samples[0].spec.shape[0])

    lengths = [sample.spec.shape[1] for sample in samples]
    miu = statistics.mean(lengths)
    std = statistics.stdev(lengths)
    low = min(lengths)
    hgh = max(lengths)
    print("Mean/std length: %.1f/%.1f" % (miu, std))
    print("Min/max length: %d/%d" % (low, hgh))

    counter = collections.Counter()
    for sample in samples:
        counter[sample.uid] += 1
    min_person_set = min(counter.values())
    print("Minimum person-specific samples:", min_person_set)
    
