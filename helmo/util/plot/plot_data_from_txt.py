import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument(
    "files",
    help="File names from which data is taken. File names are matched with labels in "
         " lexicographical order.",
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
    help="A number column in txt results which store x values",
    type=int,
    default=0,
)
parser.add_argument(
    "--y_col",
    help="A number column in txt results which store y values",
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
    help="Path to output file where plot data will be saved",
    default="plot_data.pickle",
    type=argparse.FileType('wb'),
)
args = parser.parse_args()

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

plot_data = dict()

for lbl, x, y, err in zip(args.labels, X, Y, Err):
    plot_data[lbl] = [x, y, err]

pickle.dump(plot_data, args.output)
