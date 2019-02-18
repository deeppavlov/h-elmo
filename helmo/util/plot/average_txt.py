import argparse
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    'files',
    help="A list of txt files from which lines are loaded. "
         "All files have to contain similar amounts of points. "
         "Each point have to be placed in individual line and to"
         " have the format 'x y'.",
    nargs='+',
    type=argparse.FileType('r'),
    default=None
)
parser.add_argument(
    '--output',
    help="Path to file where results will be stored. Points will have"
         " format 'x mean_y stddev_y, stderr_of_mean_y'. Default is"
         " mean.txt",
    type=argparse.FileType('w'),
    default='mean.txt'
)
parser.add_argument(
    '--preprocess',
    help="Function applied to values before averaging. Possible "
         "options: (1)sqrt. Default is None",
    default=None,
)
args = parser.parse_args()

if args.preprocess == 'sqrt':
    func = np.sqrt
else:
    func = lambda x: x


class ValuesDontMatchError(Exception):
    def __init__(self, message, target_values, tested_values):
        self.message = message
        self.target_values = target_values,
        self.tested_values = tested_values,
        self.target_len = len(self.target_values),
        self.tested_len = len(self.tested_values)


N = len(args.files)

Y = []
x_list = None

for f in args.files:
    x_l = []
    y_l = []
    for file_line in f:
        x, y = map(float, file_line.split())
        x_l.append(x)
        y_l.append(func(y))
    Y.append(y_l)
    if x_list is not None:
        if x_list != x_l:
            raise ValuesDontMatchError(
                "X values in averaged files don't match.",
                target_values=x_list,
                tested_values=x_l,
            )
    x_list = x_l

for_averaging = np.array(Y)
mean = np.mean(Y, axis=0)
std = np.std(Y, axis=0, ddof=1)
stderr_of_mean = std / N**0.5

for x, y_mean, y_std, y_stderr_of_mean in zip(x_list, mean, std, stderr_of_mean):
    args.output.write("{} {} {} {}\n".format(x, y_mean, y_std, y_stderr_of_mean))
