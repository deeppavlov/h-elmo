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
    'cross_hist',
    help="Path to pickle file with objects with"
         "cross histograms. File has to be pickle dump of"
         " numpy arrays."
)
parser.add_argument(
    '--bin_axis',
    help="The axis of bin counts for histogram.",
    type=int,
    default=0,
)
parser.add_argument(
    "--cross_bin_axis",
    help="The axis of bin counts for cross histograms.",
    type=int,
    default=0,
)
parser.add_argument(
    'output',
    help="Path to output file. Has to have '.pickle'"
         " extension."
)
args = parser.parse_args()

with open(args.hist, 'rb') as hist_f, \
        open(args.cross_hist, 'rb') as cross_f, \
        open(args.output, 'wb') as out_f:
    while True:
        try:
            hist = pickle.load(hist_f)
            cross_hist = pickle.load(cross_f)
        except EOFError:
            break
        pickle.dump(
            entropy.mutual_information_from_hist(
                hist,
                cross_hist,
                -1,
                args.bin_axis,
                args.cross_bin_axis,
                'MLE'
            ),
            out_f,
        )
