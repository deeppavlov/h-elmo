import os
import argparse
import pickle

import numpy as np

import helmo.util.plot.plot_helpers as plot_helpers


parser = argparse.ArgumentParser(
    description="Script creates a pickle file with data for drawing a plot."
                " Resulting object in pickle file is a PlotData instance. Keys of"
                " PlotData instance are labels of lines in legend, "
                "values are Line instances. Line contains keys 'x', 'y' and"
                "can contain key 'x_err' and 'y_err'."
                " LineData and PlotData implementations  are provided in plot_helpers.py."
                " X values, Y values, and errors are provided to script"
                " in separate arguments --step, --mean, and --stddev. By default lists of "
                " file names in these arguments are sorted before zipped. X values are"
                " provided in text files (validation results, e.g. loss_valid.txt)."
                " Y and Y error values are provided in pickle files, generated by "
                "average_pickle_values.py script. Figures are drawn by "
                "plot_from_pickle.py script.",
)
parser.add_argument(
    "--labels",
    "-l",
    help="Names of lines on plot. Number of labels has to match number of "
         "mean, stddev, and step files.",
    nargs='+',
)
parser.add_argument(
    "--step",
    "-s",
    help="A list of files with validation results or alternatively list"
         " of any files in which each line starts with steps.",
    nargs='+',
    type=argparse.FileType('r'),
)
parser.add_argument(
    "--mean",
    "-m",
    help="A list of pickle files with mean values",
    nargs='+',
    type=argparse.FileType('rb'),
)
parser.add_argument(
    "--stddev",
    "-d",
    help="A list of pickle files with stddev values.",
    nargs='+',
    type=argparse.FileType('rb'),
)
parser.add_argument(
    "--output",
    "-o",
    help="Path to output file where plot data will be saved",
    default="plot_data.pickle"
)
parser.add_argument(
    "--no_sort",
    "-n",
    help="Do not sort --step, --mean, --stddev before zipping.",
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
parser.add_argument(
    '--preprocess',
    "-p",
    help="Function applied to values before storing into plot_data. Possible "
         "options: (1)sqrt. Default is None",
    default=None,
)
parser.add_argument(
    '--start_idx',
    "-i",
    help="Index of the first point in plot_data. Points with indices less than "
         "start_idx are not added. Default is zero.",
    type=int,
    default=0,
)
args = parser.parse_args()

if args.preprocess == 'sqrt':
    f = np.sqrt

    def std_f(t):
        return t[0] / (2 * np.sqrt(t[1] + 1e-15))
else:
    def f(x):
        return x

    def std_f(t):
        return t[0]


if not args.no_sort:
    args.mean = sorted(args.mean, key=lambda x: x.name)
    if args.stddev is not None:
        args.stddev = sorted(args.stddev, key=lambda x: x.name)
    args.labels = sorted(args.labels)


def extract_steps_from_valid_results(file):
    return [int(line.split()[0]) for line in file]


class PickleContentIsNotDataSeries(Exception):
    def __init__(self, message):
        super().__init__(message)


def load_list_from_pickle_file(file):
    zeroth = pickle.load(file)
    try:
        first = pickle.load(file)
    except EOFError:
        try:
            iter(zeroth)
        except TypeError:
            raise PickleContentIsNotDataSeries(
                "pickle file {} contains 1 not iterable object. File has to contain\n"
                "\teither list or 1 dim array of numbers\n"
                "\tor several numbers as separate objects."
            )
        return list(zeroth)
    res = [zeroth, first]
    while True:
        try:
            res.append(pickle.load(file))
        except EOFError:
            break
    return res


means = [load_list_from_pickle_file(f) for f in args.mean]
if args.stddev is not None:
    stddevs = [load_list_from_pickle_file(f) for f in args.stddev]
else:
    stddevs = [[0.] * len(m) for m in means]

steps = [extract_steps_from_valid_results(f) for f in args.step]

output_dir = os.path.split(args.output)[0]
os.makedirs(output_dir, exist_ok=True)

plot_data = plot_helpers.PlotData()
if args.sorting_key is not None:
    exec(args.sorting_key)
    plot_data.set_sorting_key(sorting_key)
    exec_file_name = os.path.splitext(args.output)[0] + '_exec.py'
    with open(exec_file_name, 'w') as exec_f:
        exec_f.write(args.sorting_key)

for lbl, stp, mn, std in zip(args.labels, steps, means, stddevs):
    mn = list(map(f, mn))
    std = list(map(std_f, zip(std, mn)))
    # print(std)
    plot_data[lbl] = {
        'x': stp[args.start_idx:],
        'y': mn[args.start_idx:],
        'y_err': std[args.start_idx:]
    }

with open(args.output, 'wb') as f:
    pickle.dump(plot_data, f)
