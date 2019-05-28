import argparse
import pickle
import logging
import sys
import os

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
    help="A Python 3 expression passed to `eval()` built-in function."
         " Have to have following format"
         " 'func1({array}) * func2({array})'"
         "expression will be applied to numpy array",
    default=None,
)
parser.add_argument(
    '--output_postfix',
    '-p',
    help="A postfix added to output file names. If input file name is"
         " 'file.pickle' and postfix is '_postfix', output file name"
         " will be 'file_postfix.pickle'. Default is empty string ''",
    default=''
)
args = parser.parse_args()

root = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
root.addHandler(handler)
root.setLevel(logging.INFO)

for file in args.files:
    values = []
    while True:
        try:
            old_value = pickle.load(file)
            values.append(eval(args.code.format(array=old_value)))
        except EOFError:
            break
    root.info("number of values: {} - {}".format(len(values), file.name))
    file.close()

    ifn = file.name
    ofn = os.path.splitext(ifn)[0] + args.output_postfix + '.pickle'
    with open(ofn, 'wb') as of:
        for v in values:
            pickle.dump(v, of)
