import argparse
import pickle

import helmo.util.plot.plot_helpers as plot_helpers

parser = argparse.ArgumentParser()

parser.add_argument(
    "files",
    help="File names from which data is taken. File names are matched with labels in "
         " lexicographical order if --no_sort is not provided.",
    type=argparse.FileType('rb'),
    nargs='+',
)
parser.add_argument(
    "--output",
    "-o",
    help="Output file name",
    type=argparse.FileType('wb'),
)
args = parser.parse_args()

res = plot_helpers.PlotData()

for f in args.files:
    res.update(pickle.load(f))

pickle.dump(res, args.output)
