import argparse
import pickle
import logging
import sys

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
args = parser.parse_args()
# print(args.files)


root = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
root.addHandler(handler)
root.setLevel(logging.INFO)

for file in args.files:
    # print(file)
    try:
        old_value = pickle.load(file)
    except EOFError:
        old_value = None
    root.info("zeroth value: {} - {}".format(old_value, file.name))
