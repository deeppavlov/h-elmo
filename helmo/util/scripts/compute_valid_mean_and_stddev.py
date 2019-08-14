import argparse
import os
from collections import OrderedDict

import numpy as np

import helmo.util.python as python_ops


METRIC_ORDER = ['loss', 'bpc', 'perplexity', 'accuracy']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_with_launches",
        help="A path to directory with launch results. It has"
             " to contain one or more directories with non negative"
             " integers for names.",
    )
    parser.add_argument(
        "--checkpoint_subgraph_name",
        "-c",
        help="A name of subgraph from which best step is taken. "
             "Default is 'all_vars'",
        default='all_vars',
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        help="A dataset over which average is computed. Default is 'valid'",
        default='valid'
    )
    parser.add_argument(
        "--output_file_name",
        "-o",
        help="Name of output file without directory. Default is "
             "valid_mean_and_stddev.txt",
        default="valid_mean_and_stddev.txt",
    )
    parser.add_argument(
        "--output_dir",
        "-O",
        help="Directory where output is saved. By default results are saved "
             "in directory with launches."
    )
    args = parser.parse_args()
    return args


def get_metric_and_dataset_name_from_file_name(file_name):
    name, _ = os.path.splitext(file_name)
    words = name.split('_')
    return words[0], '_'.join(words[1:])


def get_metric_file_names(launch_dir, dataset_name):
    result = {}
    for fn in os.listdir(launch_dir):
        metric, dt_name = get_metric_and_dataset_name_from_file_name(fn)
        if dt_name == dataset_name:
            result[metric] = os.path.join(launch_dir, fn)
    return result


def get_best_step(launch_dir, subgraph_name):
    best_step_fn = os.path.join(launch_dir, 'checkpoints', subgraph_name, 'best_step.txt')
    with open(best_step_fn) as f:
        return int(f.read())


def extract_best_value(file_name, best_step):
    with open(file_name) as f:
        for line in f.readlines():
            step, v = line.split()
            if int(step) == best_step:
                return float(v)
    raise ValueError('specified best step is not found')


def extract_best_values(metric_file_names, best_step):
    result = {}
    for metric, fn in metric_file_names.items():
        result[metric] = extract_best_value(fn, best_step)
    return result


def get_launch_dirs(dir_with_launches):
    dirs = []
    for d in os.listdir(dir_with_launches):
        if python_ops.is_string_non_negative_int(d):
            dirs.append(os.path.join(dir_with_launches, d))
    return dirs


def get_one_launch_results(launch_dir, dataset_name, subgraph_name):
    metric_file_names = get_metric_file_names(launch_dir, dataset_name)
    best_step = get_best_step(launch_dir, subgraph_name)
    return extract_best_values(metric_file_names, best_step)


def compute_stats_from_launch_results(results_by_launches):
    results = OrderedDict([(m, []) for m in METRIC_ORDER])
    for launch_result in results_by_launches:
        for m in METRIC_ORDER:
            results[m].append(launch_result[m])
    for m, v in results.items():
        results[m] = [np.mean(v), np.std(v, ddof=1)]
    return results


def get_mean_and_stddev(dir_with_launches, dataset_name, subgraph_name):
    results_by_launches = []
    launch_dirs = get_launch_dirs(dir_with_launches)
    for launch_dir in launch_dirs:
        results_by_launches.append(get_one_launch_results(launch_dir, dataset_name, subgraph_name))
    return compute_stats_from_launch_results(results_by_launches)


def save_mean_and_stddev(results, output_dir, output_file_name):
    output_path = os.path.join(output_dir, output_file_name)
    with open(output_path, 'w') as f:
        for m, v in results.items():
            f.write('{} {} {}\n'.format(m, v[0], v[1]))


def main():
    args = get_args()
    results = get_mean_and_stddev(args.dir_with_launches, args.dataset_name, args.checkpoint_subgraph_name)
    save_mean_and_stddev(results, args.output_dir, args.output_file_name)


if __name__ == '__main__':
    main()
