import argparse
import pickle

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "file",
    help="Path to pickle file with broken histograms",
)
parser.add_argument(
    "output",
    help="Path to file with output. Has to have '.pickle'"
         " extension."
)
args = parser.parse_args()


def was_interruption(h1, h2):
    return np.logical_and(0 <= h2, h2 < h1).any()


with open(args.file, 'rb') as in_f, open(args.output, 'wb') as out_f:
    h1 = pickle.load(in_f)
    pickle.dump(h1, out_f)
    while True:
        try:
            h2 = pickle.load(in_f)
        except EOFError:
            break
        if was_interruption(h1, h2):
            pickle.dump(h2, out_f)
        else:
            pickle.dump(h2 - h1, out_f)
        h1 = h2
