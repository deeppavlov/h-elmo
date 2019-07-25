import argparse
import pickle

import helmo.util.numpy_entropy as entropy


parser = argparse.ArgumentParser()
parser.add_argument(
    "file",
    help="Path to pickle file with dumps of arrays.",
)
parser.add_argument(
    "output",
    help="Path to pickle file with mean values of"
         " given arrays."
)
parser.add_argument(
    "--axes",
    "-a",
    help="Numbers of axes which diagonal is excluded. "
         "Default is 0 and 1.",
    nargs=2,
    type=int,
    default=[0, 1],
)
args = parser.parse_args()

with open(args.file, 'rb') as in_f, open(args.output, 'wb') as out_f:
    while True:
        try:
            a = pickle.load(in_f)
        except EOFError:
            break
        pickle.dump(entropy.mean_without_diag(a, args.axes), out_f)
