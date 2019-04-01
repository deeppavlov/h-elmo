import argparse
import pickle

import helmo.util.plot.plot_helpers as plot_helpers

parser = argparse.ArgumentParser()

parser.add_argument(
    "files",
    help="File names from which data is taken. File names are matched with labels in "
         " lexicographical order if --no_sort is not provided.",
    nargs='+',
)
parser.add_argument(
    "--labels",
    help="Names of lines on plot. Number of labels has to match number of "
         "files.",
    nargs='+',
)
parser.add_argument(
    "--x_col",
    help="A number column in txt results which store x values. Default is"
         " 0.",
    type=int,
    default=0,
)
parser.add_argument(
    "--y_col",
    help="A number column in txt results which store y values. Default is 1.",
    type=int,
    default=1,
)
parser.add_argument(
    "--err_col",
    help="A number column in txt results which store y error values.",
    type=int,
    default=None,
)
parser.add_argument(
    "--output",
    "-o",
    help="Path to output file where plot data will be saved. Default is"
         " plot_data.pickle.",
    default="plot_data.pickle",
    type=argparse.FileType('wb'),
)
parser.add_argument(
    "--no_sort",
    help="Do not sort --labels and files before zipping.",
    action="store_true",
)
args = parser.parse_args()

if not args.no_sort:
    args.files = sorted(args.files)
    args.labels = sorted(args.labels)


X = []
Y = []
Err = []

for fn in args.files:
    with open(fn) as f:
        x = []
        y = []
        err = []
        for l in f:
            values = list(map(float, l.split()))
            x.append(values[args.x_col])
            y.append(values[args.y_col])
            if args.err_col is not None:
                err.append(values[args.err_col])
            else:
                err.append(0.)
        X.append(x)
        Y.append(y)
        Err.append(err)

plot_data = plot_helpers.PlotData()

for lbl, x, y, err in zip(args.labels, X, Y, Err):
    plot_data[lbl] = {'x': x, 'y': y, 'y_err': err}

pickle.dump(plot_data, args.output)
