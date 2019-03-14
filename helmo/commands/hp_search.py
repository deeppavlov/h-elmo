import sys
import os
import json
import tensorflow as tf
import argparse

sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/h-elmo')),
    os.path.expanduser('~/h-elmo'),
    os.path.join('/cephfs', os.path.expanduser('~/repos/learning-to-learn')),
    os.path.expanduser('~/repos/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/repos/h-elmo')),
    os.path.expanduser('~/repos/h-elmo'),
    '/cephfs/home/peganov/learning-to-learn',
    '/home/peganov/learning-to-learn',
    '/cephfs/home/peganov/h-elmo',
    '/home/peganov/h-elmo',
]

import helmo.util.dataset
import helmo.util.import_help
import helmo.util.path_help

from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary, \
    compose_hp_confs, get_num_exps_and_res_files

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

# save_path_relative_to_expres = os.path.join(*config_path.split('.')[:-1])
# # print(save_path_relative_to_expres)
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
save_path = helmo.util.path_help.get_save_path_from_config_path(
    config_path, dir_with_confs, results_directory_rel_to_repo_root)
results_file_name = os.path.join(save_path, 'test.txt')
confs, _ = compose_hp_confs(
    config_path, results_file_name, chop_last_experiment=False, model='pupil')
confs.reverse()  # start with small configs
print("confs:", confs)

text = helmo.util.dataset.get_text(config['dataset']['path'])
test_size = int(config['dataset']['test_size'])
valid_size = int(config['dataset']['valid_size'])
train_size = len(text) - test_size - valid_size
test_text, valid_text, train_text = helmo.util.dataset.split_text(text, test_size, valid_size, train_size)

vocabulary, vocabulary_size = helmo.util.dataset.get_vocab(config['dataset']['vocab_path'], text)

env = Environment(Net, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

evaluation = config['evaluation'].copy()
evaluation['save_path'] = save_path
evaluation['datasets'] = [(test_text, 'test')]
evaluation['batch_gen_class'] = BatchGenerator
evaluation['batch_kwargs']['vocabulary'] = vocabulary
if 'additional_feed_dict' not in evaluation:
    evaluation['additional_feed_dict'] = []
kwargs_for_building = config["build"].copy()
kwargs_for_building['voc_size'] = vocabulary_size
launch_kwargs = config['launch'].copy()
launch_kwargs['train_dataset_text'] = train_text
launch_kwargs['vocabulary'] = vocabulary
launch_kwargs['validation_datasets'] = {'valid': valid_text}
if 'restore_path' in launch_kwargs:
    launch_kwargs['restore_path'] = helmo.util.path_help.prepend_restore_path_with_expres(launch_kwargs['restore_path'])

if config['seed'] is not None:
    tf.set_random_seed(config['seed'])

for conf in confs:
    build_hyperparameters = dict()
    for name in config['build_hyperparameters']:
        build_hyperparameters[name] = conf[name]
    other_hyperparameters = dict()
    for name, tmpl in config['other_hyperparameters'].items():
        main_name = name.split('[')[0]
        tmpl = tmpl.copy()
        if 'varying' in tmpl:
            for param, values in conf.items():
                if main_name in param:
                    if isinstance(tmpl['varying'], dict):
                        tmpl['varying'][param.split('/')[-1]] = values
                    else:
                        tmpl['varying'] = values
            other_hyperparameters[name] = tmpl
        else:
            other_hyperparameters[name] = conf[name]
    # print("(hp_search)other_hyperparameters:", other_hyperparameters)
    _, biggest_idx, _ = get_num_exps_and_res_files(save_path)
    if biggest_idx is None:
        initial_experiment_counter_value = 0
    else:
        initial_experiment_counter_value = biggest_idx + 1
    # print("(hp_search)build_hyperparameters:", build_hyperparameters)
    # print("(hp_search)other_hyperparameters:", other_hyperparameters)
    env.grid_search(
        evaluation,
        kwargs_for_building,
        build_hyperparameters=build_hyperparameters,
        other_hyperparameters=other_hyperparameters,
        initial_experiment_counter_value=initial_experiment_counter_value,
        **launch_kwargs,
    )
