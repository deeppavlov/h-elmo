import argparse
import pickle

import helmo.util.numpy_entropy as entropy


parser = argparse.ArgumentParser()
parser.add_argument(
    'hist',
    help="Path to pickle file with objects with"
         "histograms. File has to be pickle dump of"
         " numpy arrays."
)
parser.add_argument(
    '--bin_axis',
    help="The axis of bin counts.",
    type=int,
    default=0,
)
parser.add_argument(
    'output',
    help="Path to output file. Has to have '.pickle'"
         " extension."
)
args = parser.parse_args()

with open(args.hist, 'rb') as in_f, open(args.output, 'wb') as out_f:
    while True:
        try:
            obj = pickle.load(in_f)
        except EOFError:
            break
        pickle.dump(entropy.entropy_MLE_from_hist_numpy(obj, args.bin_axis), out_f)
