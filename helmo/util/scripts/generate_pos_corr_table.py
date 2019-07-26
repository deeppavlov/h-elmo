import argparse
import os
import pickle

import helmo.util.formatting as formatting


def get_collected_tags(mean_dir):
    tags = set()
    for layer_name in [e for e in os.listdir(mean_dir) if os.path.isdir(os.path.join(mean_dir, e))]:
        path = os.path.join(mean_dir, layer_name)
        tags |= set(
            [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
        )
    return list(tags)


def read_tags(file_name):
    tags = []
    with open(file_name) as f:
        for line in f.readlines():
            tags.append(line.strip())
    return tags


def get_tags(mean_dir, file_with_tag_order):
    if file_with_tag_order is None:
        return sorted(get_collected_tags(mean_dir))
    return read_tags(file_with_tag_order)


def get_file_by_prefix(path, prefix):
    for fn in os.listdir(path):
        if fn.startswith(prefix):
            return os.path.join(path, fn)


def get_value_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)[0]


def get_values(path, error_type):
    mean_file = get_file_by_prefix(path, 'mean_')
    error_file = get_file_by_prefix(path, error_type + '_')
    mean = get_value_from_pickle(mean_file)
    error = get_value_from_pickle(error_file)
    return mean, error


def fill_field(results, path, i, j, error_type, std_acc):
    mean, error = get_values(path, error_type)
    mean, error, nd = formatting.round_mean_and_std(mean, error, std_acc)
    results[i][j] = formatting.create_plus_minus_str(mean, error, nd)


def save_table_text(results, file_name, delim):
    with open(file_name, 'w') as f:
        for row in results:
            f.write(delim.join(row) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mean_dir",
        help="Directory with mean results by layers. First level dirs"
             " have to be named  after layers, second level directories "
             "have to be named after tags. Second level dirs have to contain"
             "three pickle files each. This files include mean, stddev and"
             "stderr_of_mean files. File names have start with 'mean_', "
             "'stddev_', 'stderr_of_mean_' correspondingly."
    )
    parser.add_argument(
        "output",
        help="Path to file table text."
    )
    parser.add_argument(
        "--acc",
        help="Relative accuracy of stddev computation. Based on accuracy stddev and"
             " mean rounding is performing.",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--order",
        help="Path to .txt file where the order of tags in table is specified."
             "Tags have to be provided on individual lines. If None tags are"
             "sorted in alphabetical order."
    )
    parser.add_argument(
        "--error_type",
        help="You choose what type of error you wish to add to table. Possible"
             " options are `stddev` and `stderr_of_mean`. Default is `stddev`.",
        default='stddev',
    )
    parser.add_argument(
        "--delim",
        help="A character used for separation of values in one row of a table. "
             "Default is semicolon.",
        default=';',

    )
    args = parser.parse_args()

    layers = [e for e in sorted(os.listdir(args.mean_dir)) if os.path.isdir(os.path.join(args.mean_dir, e))]
    tags = get_tags(args.mean_dir, args.order)
    results = [[None]*len(layers) for _ in tags]
    for j, layer in enumerate(layers):
        for i, tag in enumerate(tags):
            path = os.path.join(args.mean_dir, layer, tag)
            if os.path.exists(path):
                fill_field(results, path, i, j, args.error_type, args.acc)
    save_table_text(results, args.output, args.delim)
