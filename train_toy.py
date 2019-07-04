import argparse

import voxceleb1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker_id_config_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    args = parser.parse_args()

    voxceleb1.train_toy(
        args.speaker_id_config_path,
        args.log_path,
        args.num_samples
    )
