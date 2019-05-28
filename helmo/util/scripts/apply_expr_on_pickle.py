import argparse
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    'files',
    help="A list of pickle files from which lines are loaded. "
         "All files have to contain similar amounts of objects "
         "which have to have similar shapes.",
    nargs='+',
    type=argparse.FileType('rb'),
    default=None
)
parser.add_argument(
    '--code',
    '-c',
    help="A Python 3 expression passed to `eval()` built-in function. "
         " Have to have following format "
         " 'function({0}) * {1}**{0}' "
         "expression will be applied to numpy arrays where indices are "
         "numbers of files in which arrays are stored. ",
    default=None,
)
parser.add_argument(
    '--output',
    '-o',
    help="An output pickle file",
    default='output.pickle',
)
args = parser.parse_args()


arrays = []
for file in args.files:
    arrays.append(pickle.load(file))

result = eval(args.code.format(*args.files))

with open(args.output) as f:
    pickle.dump(result, f)
