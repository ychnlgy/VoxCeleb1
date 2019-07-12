import argparse

import voxceleb1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--modelf", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--embed_size", type=int, required=True)
    args = parser.parse_args()
    
    voxceleb1.training.Config.write(vars(args), path_header="path")
    print("Successfully wrote to %s" % args.path)
