import argparse

import voxceleb1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    
    config = voxceleb1.training.Config(args.path)
    del config.param_dict["_dob"]
    kvs = ["--%s %s" % item for item in config.param_dict.items()]
    print(" ".join(kvs))
