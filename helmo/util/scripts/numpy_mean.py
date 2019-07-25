import argparse
import pickle

import numpy as np


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
args = parser.parse_args()

with open(args.file, 'rb') as in_f, open(args.output, 'wb') as out_f:
    while True:
        try:
            a = pickle.load(in_f)
        except EOFError:
            pickle.dump(np.mean(a))
