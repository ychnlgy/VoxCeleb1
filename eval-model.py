import argparse

import voxceleb1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--veri_test_path", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--stat_path", required=True)
    parser.add_argument("--speaker_id_config_path", required=True)
    parser.add_argument("--speaker_dist_config_path", required=True)
    parser.add_argument("--use_embedding", type=int, required=True)
    parser.add_argument("--min_samples", type=int, required=True)
    args = parser.parse_args()

    voxceleb1.testing.main(
        veri_test_path=args.veri_test_path,
        dataroot=args.dataroot,
        stat_path=args.stat_path,
        speaker_id_config_path=args.speaker_id_config_path,
        speaker_dist_config_path=args.speaker_dist_config_path,
        use_embedding=args.use_embedding
    )
