import argparse

from matplotlib import pyplot

from voxceleb1 import load

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    samples = list(load(args.file))

    try:
        for sample in samples:
            print(sample.dev, sample.uid, sample.spec.shape)

        for sample in samples:
            pyplot.imshow(sample.spec[:,:300])
            pyplot.show()
    except KeyboardInterrupt:
        pass
