import os
import json
import multiprocessing as mp
import argparse

import tensorflow as tf

import helmo.util.dataset
import helmo.util.import_help
import helmo.util.path_help
import helmo.util.results
from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary, \
    compose_hp_confs, get_num_exps_and_res_files, compute_stddev


parser = argparse.ArgumentParser()
parser.add_argument(
    "config_path",
    help="Path to JSON config of experiment."
)
parser.add_argument(
    "--test",
    help="Use script for testing. In this case results of testing are put into"
         " directory testres in repo root.",
    action='store_true',
)
parser.add_argument(
    "--no_logging",
    help="If provided tf logs are not shown. Errors are still shown.",
    action="store_true",
)
args = parser.parse_args()

if args.no_logging:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config_path = args.config_path

with open(config_path) as f:
    config = json.load(f)

exec(helmo.util.import_help.form_load_cmd(
    config['batch_gen']['path'], config['batch_gen']['cls_name'], "BatchGenerator"))
exec(helmo.util.import_help.form_load_cmd(config['net']['path'], config['net']['cls_name'], "Net"))


dir_with_confs, results_directory_rel_to_repo_root = \
    ('tests', 'testres') if args.test else ('experiments', 'expres')

save_path = helmo.util.path_help.get_save_path_from_config_path(
    config_path, dir_with_confs, results_directory_rel_to_repo_root)

metrics, _, finished_launches = helmo.util.results.load_tt_results(
    save_path, config['test']['result_types'])

test_datasets, _, _ = helmo.util.dataset.get_datasets_using_config(config['dataset'])

vocabulary, vocabulary_size = helmo.util.dataset.get_vocab(
    config['dataset']['vocab_path'],
    helmo.util.dataset.get_text(config['dataset']['path']),
)

env = Environment(Net, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

kwargs_for_building = config["graph"]
kwargs_for_building['voc_size'] = vocabulary_size

test_kwargs = config['test']
test_kwargs['vocabulary'] = vocabulary
test_kwargs['validation_datasets'] = test_datasets


def compose_restore_paths(rp_conf):
    result = []
    for path in rp_conf['paths']:
        if 'prefix' in rp_conf:
            path = os.path.join(rp_conf['prefix'], path)
        if 'postfix' in rp_conf:
            path = os.path.join(path, rp_conf['postfix'])
        result.append(path)
    return result


restore_paths = compose_restore_paths(config['restore_paths'])


def test(q, launch_folder, restore_path):
    print('testing')
    test_kwargs['save_path'] = os.path.join(save_path, launch_folder, 'testing')
    test_kwargs['restore_path'] = restore_path
    env.build_pupil(**kwargs_for_building)
    _, _, mean_metrics = env.test(**test_kwargs)
    description_file = os.path.join(test_kwargs['save_path'], 'description.txt')
    with open(description_file, 'w') as f:
        f.write('restore_path:\n{}\n'.format(restore_path))
    q.put(mean_metrics)


n = len(restore_paths)
for i in range(n):
    launch_folder = str(i)
    if str(i) not in finished_launches:
        try:
            restore_path = restore_paths[int(launch_folder)]
        except KeyError:
            continue
        print("LAUNCH NUMBER %s" % launch_folder)
        print('RESTORING FROM {}'.format(restore_path))
        q = mp.Queue()
        p = mp.Process(target=test, args=(q, launch_folder, restore_path))
        p.start()
        metrics.append(q.get()['test'])
        p.join()


keys = list(metrics[0].keys())
metrics_ = {key: list() for key in keys}
for launch_res in metrics:
    for key in keys:
        metrics_[key].append(launch_res[key])

if config['calculate_mean']:
    with open(os.path.join(save_path, 'test_mean_and_stddev.txt'), 'w') as f:
        for key, metric_values in metrics_.items():
            f.write("%s %s %s\n" % (key, sum(metric_values) / len(metric_values), compute_stddev(metric_values)))
