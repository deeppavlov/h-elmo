import argparse
import os
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
    "-l",
    help="Names of lines on plot. Number of labels has to match number of "
         "files.",
    nargs='+',
)
parser.add_argument(
    "--x_col",
    "-x",
    help="A number column in txt results which store x values. Default is"
         " 0.",
    type=int,
    default=0,
)
parser.add_argument(
    "--y_col",
    "-y",
    help="A number column in txt results which store y values. Default is 1.",
    type=int,
    default=1,
)
parser.add_argument(
    "--err_col",
    "-e",
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
)
parser.add_argument(
    "--no_sort",
    "-n",
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
         "ATTENTION. anonymous functions are not supported. "
         "The sorting key code is stored in the file with name "
         "of the form {args.output}_exec.py. For plotting with "
         "script 'plot_from_pickle.py' you need to pass path to "
         "sorting key script in parameter -e."
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

output_dir = os.path.split(args.output)[0]
os.makedirs(output_dir, exist_ok=True)

if args.sorting_key is not None:
    exec(args.sorting_key)
    plot_data.set_sorting_key(sorting_key)

    exec_file_name = os.path.splitext(args.output)[0] + '_exec.py'
    with open(exec_file_name, 'w') as exec_f:
        exec_f.write(args.sorting_key)
for lbl, x, y, err in zip(args.labels, X, Y, Err):
    plot_data[lbl] = {'x': x, 'y': y, 'y_err': err}

with open(args.output, 'wb') as f:
    pickle.dump(plot_data, f)
