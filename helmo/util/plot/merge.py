import argparse
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    'xsrc',
    help="File with X values. It can .txt or .pickle. If it is .txt"
         " colx have to be provided",
)
parser.add_argument(
    'ysrc',
    help="File with Y values. It can .txt or .pickle. If it is .txt"
         " coly have to be provided",
)
parser.add_argument(
    '--colx',
    help="Number of a column in srcx if it is text file.",
    type=int,
)
parser.add_argument(
    '--coly',
    help="Number of a column in srcy if it is text file.",
    type=int,
)
parser.add_argument(
    '--xerrsrc',
    help="File with X error values. Have to be provided if xsrc is pickle",
)
parser.add_argument(
    '--yerrsrc',
    help="File with X error values. Have to be provided if ysrc is pickle",
)
parser.add_argument(
    '--xerrcol',
    help="Number of X error column in xsrc if xsrc is txt file",
    type=int,
)
parser.add_argument(
    '--yerrcol',
    help="Number of X error column in ysrc if ysrc is txt file",
    type=int,
)
parser.add_argument(
    '--output',
    help="Path to file where results will be stored. Points will have"
         " format 'x mean_y stddev_y, stderr_of_mean_y'. Default is"
         " merge.txt",
    type=argparse.FileType('w'),
    default='merge.txt'
)
parser.add_argument(
    '--reverse',
    help="If provided points are sorted in reverse order.",
    action='store_true',
)

args = parser.parse_args()


def fill_values_and_err(src, col, errsrc, errcol):
    if src[-4:] == '.txt':
        v = []
        err = []
        with open(src) as f:
            for l in f:
                values = [float(v) for v in l.split()]
                v.append(values[col])
                if errcol is not None:
                    err.append(values[errcol])
                else:
                    err.append(0.)
    else:
        with open(src, 'rb') as f:
            v = pickle.load(f)
        if errsrc is None:
            err = [0.]*len(v)
        else:
            with open(errsrc, 'rb') as f:
                err = pickle.load(f)
    return v, err


x, xerr = fill_values_and_err(args.xsrc, args.colx, args.xerrsrc, args.xerrcol)
y, yerr = fill_values_and_err(args.ysrc, args.coly, args.yerrsrc, args.yerrcol)

for xx, yy, xxerr, yyerr in sorted(zip(x, y, xerr, yerr), key=lambda x: x[0], reverse=args.reverse):
    args.output.write('{} {} {} {}\n'.format(xx, yy, xxerr, yyerr))
