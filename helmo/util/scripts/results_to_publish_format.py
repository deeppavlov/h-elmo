import argparse
import os

import helmo.util.formatting as formatting


parser = argparse.ArgumentParser()
parser.add_argument(
    "file",
    help="File with input data. It may"
         "have zero or more lines. Last 2 words of"
         " each line are metric mean and "
         "metric stddev. Other words in line"
         " compose metric name.",
)
parser.add_argument(
    "--output",
    "-o",
    help="Path to file with script output. If not provided "
         "results are save in the same directory with inputs "
         "in file with name 'publish_format_results.txt'"
)
args = parser.parse_args()

if args.output is None:
    path = os.path.split(args.file)[0]
    out_fn = os.path.join(path, 'publish_format_results.txt')
else:
    out_fn = args.output

with open(args.file) as in_f, open(out_fn, 'w') as out_f:
    for line in in_f.readlines():
        words = line.split()
        mean = float(words[-2])
        stddev = float(words[-1])
        name = ' '.join(words[:-2])
        out_f.write(
            name + ' ' +
            formatting.create_plus_minus_str_full(mean, stddev, 0.2) +
            '\n'
        )
