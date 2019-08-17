import argparse
import os
import pickle

import helmo.util.plot.plot_helpers as plot_helpers

parser = argparse.ArgumentParser()
parser.add_argument(
    '--xsrc',
    help="Files with X values. They can be .txt or .pickle. If they are .txt"
         " colx has to be provided",
    nargs='+',
    default=None,
)
parser.add_argument(
    '--ysrc',
    help="Files with Y values. They can .txt or .pickle. If they is .txt"
         " coly has to be provided",
    nargs='+',
    default=None,
)
parser.add_argument(
    '--colx',
    help="Number of a column in srcx if it is text file. Numbering is "
         "starting with 0",
    type=int,
)
parser.add_argument(
    '--coly',
    help="Number of a column in srcy if it is text file.",
    type=int,
)
parser.add_argument(
    '--xerrsrc',
    help="Files with X error values. Have to be provided if xsrc is pickle",
    nargs='+',
    default=None,
)
parser.add_argument(
    '--yerrsrc',
    help="Files with X error values. Have to be provided if ysrc is pickle",
    nargs='+',
    default=None,
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
    '--labels',
    help="labels of the lines",
    nargs='+',
)
parser.add_argument(
    '--output',
    '-o',
    help="Path to file where results will be stored. Points will have"
         " format 'x mean_y stddev_y, stderr_of_mean_y'. Default is"
         " merge.pickle",
    type=argparse.FileType('wb'),
    default='merge.pickle'
)
parser.add_argument(
    '--reverse',
    help="If provided points are sorted in reverse order.",
    action='store_true',
)
parser.add_argument(
    "--no_sort",
    help="Do not sort --labels and files before zipping.",
    action="store_true",
)
parser.add_argument(
    "--sorting_key",
    "-k",
    help="A function used for sorting lines before plotting. "
         "A function returns a value used for comparing labels "
         "and works the same way as `key` parameter in built-in "
         "`sorted()` function. Function passed as a string "
         "containing definition of function with name "
         "'sorting_key'. For instance \n"
         "-k 'def sorting_key(x):\n\treturn int(x[-1])'\n\n"
         "ATTENTION. anonymous functions are not supported."
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


if not args.no_sort:
    args.xsrc.sort()
    args.ysrc.sort()

if args.xerrsrc is None:
    args.xerrsrc = [None]*len(args.xsrc)
else:
    if not args.no_sort:
        args.xerrsrc.sort()

if args.yerrsrc is None:
    args.yerrsrc = [None]*len(args.ysrc)
else:
    if not args.no_sort:
        args.yerrsrc.sort()

plot_data_init = []
# print(len(args.labels))
# print(len(args.xsrc))
# print(len(args.xerrsrc))
# print(len(args.yerrsrc))
for lbl, xsrc, ysrc, xerrsrc, yerrsrc in zip(args.labels, args.xsrc, args.ysrc, args.xerrsrc, args.yerrsrc):
    x, x_err = fill_values_and_err(xsrc, args.colx, xerrsrc, args.xerrcol)
    y, y_err = fill_values_and_err(ysrc, args.coly, yerrsrc, args.yerrcol)
    x, y, x_err, y_err = zip(*sorted(zip(x, y, x_err, y_err), key=lambda x: x[0], reverse=args.reverse))
    plot_data_init.append((lbl, dict(x=x, y=y, x_err=x_err, y_err=y_err)))

pd = plot_helpers.PlotData(plot_data_init)
if args.sorting_key is not None:
    exec(args.sorting_key)
    pd.set_sorting_key(sorting_key)

    exec_file_name = os.path.splitext(args.output.name)[0] + '_exec.py'
    with open(exec_file_name, 'w') as exec_f:
        exec_f.write(args.sorting_key)

pickle.dump(
    pd,
    args.output,
)
