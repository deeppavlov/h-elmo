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
args = parser.parse_args()

config_path = args.config_path

with open(config_path) as f:
    config = json.load(f)

exec(helmo.util.import_help.form_load_cmd(config['batch_gen']['path'], config['batch_gen']['cls_name'], "BatchGenerator"))
exec(helmo.util.import_help.form_load_cmd(config['net']['path'], config['net']['cls_name'], "Net"))

# save_path_relative_to_expres = '.'.join(config_path.split('.')[:-1])
# # print(save_path_relative_to_expres)
#
# results_dir = os.path.join(
#     helmo.util.path_help.move_path_postfix_within_repo(
#         path_to_smth_in_separator=save_path_relative_to_expres,
#         separator="experiments",
#         new_prefix_within_repo="expres",
#     ),
#     os.path.split(save_path_relative_to_expres)[-1]
# )
# # print(results_dir)
# save_path = results_dir

dir_with_confs, results_directory_rel_to_repo_root = \
    ('tests', 'testres') if args.test else ('experiments', 'expres')
# print("(train_test)config_path:", config_path)
# print("(train_test)dir_with_confs:", dir_with_confs)
# print("(train_test)results_directory_rel_to_repo_root:", results_directory_rel_to_repo_root)
save_path = helmo.util.path_help.get_save_path_from_config_path(
    config_path, dir_with_confs, results_directory_rel_to_repo_root)

# print("(train_test)save_path:", save_path)

metrics, launches_for_testing, trained_launches = helmo.util.results.load_tt_results(
    save_path, config['test']['result_types'])
# print("(run_train_test)metrics:", metrics)
# print("(run_train_test)launches_for_testing:", launches_for_testing)
# print("(run_train_test)trained_launches:", trained_launches)


test_datasets, valid_datasets, train_dataset = helmo.util.dataset.get_datasets_using_config(config['dataset'])

vocabulary, vocabulary_size = helmo.util.dataset.get_vocab(
    config['dataset']['vocab_path'],
    train_dataset['train'],
)

env = Environment(Net, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

kwargs_for_building = config["graph"]
kwargs_for_building['voc_size'] = vocabulary_size

train_kwargs = config['train']
train_kwargs['train_dataset_text'] = train_dataset['train']
train_kwargs['vocabulary'] = vocabulary
train_kwargs['validation_datasets'] = valid_datasets

test_kwargs = config['test']
test_kwargs['vocabulary'] = vocabulary
test_kwargs['validation_datasets'] = test_datasets
if 'restore_path' not in test_kwargs:
    checkpoint_appendix = 'checkpoints/all_vars/best' if 'subgraphs_to_save' in train_kwargs else 'checkpoints/best'


def test(q, launch_folder):
    test_kwargs['save_path'] = os.path.join(save_path, launch_folder, 'testing')
    test_kwargs['restore_path'] = os.path.join(save_path, launch_folder, checkpoint_appendix)
    env.build_pupil(**kwargs_for_building)
    _, _, mean_metrics = env.test(**test_kwargs)
    q.put(mean_metrics)


def train(q, launch_folder):
    if config['seed'] is not None:
        tf.set_random_seed(config['seed'])
    test_kwargs['save_path'] = os.path.join(save_path, launch_folder, 'testing')
    train_kwargs['save_path'] = os.path.join(save_path, launch_folder)
    test_kwargs['restore_path'] = os.path.join(train_kwargs['save_path'], checkpoint_appendix)
    print("(run_train_test)test_kwargs['restore_path']:", test_kwargs['restore_path'])
    if os.path.isfile(test_kwargs['restore_path'] + '.index'):
        # print("(run_train_test)setting restore_path")
        train_kwargs['restore_path'] = test_kwargs['restore_path']
    env.build_pupil(**kwargs_for_building)
    env.train(**train_kwargs)
    _, _, mean_metrics = env.test(**test_kwargs)
    q.put(mean_metrics)


for launch_folder in launches_for_testing:
    print("LAUNCH NUMBER %s" % launch_folder)
    q = mp.Queue()
    p = mp.Process(target=test, args=(q, launch_folder))
    p.start()
    metrics.append(q.get()['test'])
    p.join()


for launch_folder in sorted(set(map(str, range(config['num_repeats']))) - set(trained_launches), key=int):
    print("LAUNCH NUMBER %s" % launch_folder)
    q = mp.Queue()
    p = mp.Process(target=train, args=(q, launch_folder))
    p.start()
    metrics.append(q.get()['test'])
    p.join()

keys = list(metrics[0].keys())
metrics_ = {key: list() for key in keys}
for launch_res in metrics:
    for key in keys:
        # print('(run_train_test)launch_res:', launch_res)
        # print('(run_train_test)key:', key)
        # print('(run_train_test)metrics_:', metrics_)
        metrics_[key].append(launch_res[key])

with open(os.path.join(save_path, 'test_mean_and_stddev.txt'), 'w') as f:
    for key, metric_values in metrics_.items():
        f.write("%s %s %s\n" % (key, sum(metric_values) / len(metric_values), compute_stddev(metric_values)))
