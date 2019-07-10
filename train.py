import argparse

import voxceleb1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker_id_config_path", required=True)
    parser.add_argument("--speaker_dist_config_path", required=True)
    parser.add_argument("--stat_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--max_chunks", type=int)
    parser.add_argument("--email")
    args = parser.parse_args()

    voxceleb1.train(
        args.speaker_id_config_path,
        args.speaker_dist_config_path,
        args.stat_path,
        args.data_path,
        args.log_path,
        args.cores,
        args.max_chunks,
        args.email
    )
