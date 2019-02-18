import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument(
    "--labels",
    help="Names of lines on plot. Number of labels has to match number of "
         "mean, stddev, and step files.",
    nargs='+',
)
parser.add_argument(
    "--step",
    help="A list of files with validation results or alternatively list"
         " of any file in which each line starts with steps.",
    nargs='+',
    type=argparse.FileType('r'),
)
parser.add_argument(
    "--mean",
    help="A list of pickle files with mean values",
    nargs='+',
    type=argparse.FileType('rb'),
)
parser.add_argument(
    "--stddev",
    help="A list of pickle files with stddev values. The order has to be the same as"
         " in --mean argument.",
    nargs='+',
    type=argparse.FileType('rb'),
)
parser.add_argument(
    "--output",
    help="Path to output file where plot data will be saved",
    default="plot_data.pickle"
)
args = parser.parse_args()

args.mean = sorted(args.mean, key=lambda x: x.name)
args.stddev = sorted(args.stddev, key=lambda x: x.name)
args.labels = sorted(args.labels)


def extract_steps_from_valid_results(file):
    return [int(line.split()[0]) for line in file]


def load_list_from_pickle_file(file):
    return list(pickle.load(file))


means = [load_list_from_pickle_file(f) for f in args.mean]
if args.stddev is not None:
    stddevs = [load_list_from_pickle_file(f) for f in args.stddev]
else:
    stddevs = [[0.] * len(m) for m in means]

steps = [extract_steps_from_valid_results(f) for f in args.step]

plot_data = dict()
for lbl, stp, mn, std in zip(args.labels, steps, means, stddevs):
    plot_data[lbl] = [stp, mn, std]

with open(args.output, 'wb') as f:
    pickle.dump(plot_data, f)
