import sys
import os
import json
import tensorflow as tf
sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/h-elmo')),
    os.path.expanduser('~/h-elmo'),
]


from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary, \
    compose_hp_confs, get_num_exps_and_res_files

import helmo.util.organise as organise

config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)

exec(organise.form_load_cmd(config['batch_gen']['path'], config['batch_gen']['cls_name'], "BatchGenerator"))
exec(organise.form_load_cmd(config['net']['path'], config['net']['cls_name'], "Net"))

save_path_relative_to_expres = os.path.join(*config_path.split('.')[:-1])
print(save_path_relative_to_expres)
results_dir = os.path.join(
    organise.get_path_to_dir_with_results(save_path_relative_to_expres),
    os.path.split(save_path_relative_to_expres)[-1]
)
print(results_dir)
save_path = results_dir
results_file_name = os.path.join(save_path, 'valid.txt')
confs, _ = compose_hp_confs(
    config_path, results_file_name, chop_last_experiment=False, model='pupil')
confs.reverse()  # start with small configs
print("confs:", confs)

text = organise.get_text(config['dataset']['path'])
test_size = int(config['dataset']['test_size'])
valid_size = int(config['dataset']['valid_size'])
train_size = len(text) - test_size - valid_size
test_text, valid_text, train_text = organise.split_text(text, test_size, valid_size, train_size)

vocabulary, vocabulary_size = organise.get_vocab(config['dataset']['vocab_path'], text)

env = Environment(Net, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

evaluation = config['evaluation'].copy()
evaluation['save_path'] = save_path
evaluation['datasets'] = [(valid_text, 'valid')]
evaluation['batch_gen_class'] = BatchGenerator
evaluation['batch_kwargs']['vocabulary'] = vocabulary
evaluation['additional_feed_dict'] = []
kwargs_for_building = config["kwargs_for_building"]
kwargs_for_building['voc_size'] = vocabulary_size
launch_kwargs = config['launch_kwargs']
launch_kwargs['train_dataset_text'] = train_text
launch_kwargs['vocabulary'] = vocabulary


for conf in confs:
    build_hyperparameters = dict()
    for name in config['build_hyperparameters']:
        build_hyperparameters[name] = conf[name]
    other_hyperparameters = dict()
    for name in config['other_hyperparameters']:
        other_hyperparameters[name] = conf[name]
    _, biggest_idx, _ = get_num_exps_and_res_files(save_path)
    if biggest_idx is None:
        initial_experiment_counter_value = 0
    else:
        initial_experiment_counter_value = biggest_idx + 1
    env.grid_search(
        evaluation,
        config['kwargs_for_building'],
        build_hyperparameters=build_hyperparameters,
        other_hyperparameters=other_hyperparameters,
        initial_experiment_counter_value=initial_experiment_counter_value,
        **config['launch_kwargs']
    )
