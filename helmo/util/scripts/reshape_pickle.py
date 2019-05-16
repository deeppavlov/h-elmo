import pickle
import argparse

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "path",
    help="Path to pickle file with array like objects of "
         "compatible shapes.",
    type=argparse.FileType(mode='rb'),
)
parser.add_argument(
    "--shape",
    "-s",
    help="Numpy style shape. Dims are separated with space"
         "brackets are not used, e.g. `10 10`.",
    nargs='+',
    default="auto"
)
parser.add_argument(
    "output",
    help="Path to output file.",
    type=argparse.FileType(mode='wb'),
)
args = parser.parse_args()

arrays = []
while True:
    try:
        arrays.append(pickle.load(args.path))
    except EOFError:
        break

if args.shape == 'auto':
    shape = [-1, arrays[0].shape[-1]]
else:
    shape = list(map(int, args.shape))

result = np.reshape(arrays, shape)

pickle.dump(result, args.output)
