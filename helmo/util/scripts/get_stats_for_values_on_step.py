import argparse
import json
import sys

import numpy as np

import helmo.util.scripts.get_value_on_step as get_value


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_names",
        nargs='+',
        help="Paths to files from which_values are taken",
    )
    parser.add_argument(
        "step",
        help="Step from which results are taken.",
        type=int,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file name. If not provided results are printed to sys.stdout",
        type=argparse.FileType(mode='w'),
        default=sys.stdout,
    )
    return parser.parse_args()


def get_stats(values):
    return dict(
        mean=np.mean(values),
        std=np.std(values, ddof=1),
        min=np.min(values),
        max=np.max(values),
    )


def main():
    args = get_args()
    values = [get_value.get_value_on_step(fn, args.step) for fn in args.file_names]
    stats = get_stats(values)
    json.dump(stats, args.output, indent=2)


if __name__ == '__main__':
    main()
