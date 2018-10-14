import sys
import os
import json
import multiprocessing as mp

import tensorflow as tf
sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/h-elmo')),
    os.path.expanduser('~/h-elmo'),
]


from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary, \
    compose_hp_confs, get_num_exps_and_res_files, compute_stddev

import helmo.util.organise as organise

config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)

exec(organise.form_load_cmd(config['batch_gen']['path'], config['batch_gen']['cls_name'], "BatchGenerator"))
exec(organise.form_load_cmd(config['net']['path'], config['net']['cls_name'], "Net"))

save_path_relative_to_expres = os.path.join(*config_path.split('.')[:-1])
# print(save_path_relative_to_expres)
results_dir = os.path.join(
    organise.get_path_to_dir_with_results(save_path_relative_to_expres),
    os.path.split(save_path_relative_to_expres)[-1]
)
# print(results_dir)
save_path = results_dir


text = organise.get_text(config['dataset']['path'])
test_size = int(config['dataset']['test_size'])
valid_size = int(config['dataset']['valid_size'])
train_size = len(text) - test_size - valid_size
test_text, valid_text, train_text = organise.split_text(text, test_size, valid_size, train_size)

vocabulary, vocabulary_size = organise.get_vocab(config['dataset']['vocab_path'], text)

env = Environment(Net, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

kwargs_for_building = config["graph"]
kwargs_for_building['voc_size'] = vocabulary_size

train_kwargs = config['train']
train_kwargs['train_dataset_text'] = train_text
train_kwargs['vocabulary'] = vocabulary
train_kwargs['validation_datasets'] = dict(valid=valid_text)
train_kwargs['save_path'] = save_path

test_kwargs = config['test']
test_kwargs['vocabulary'] = vocabulary
test_kwargs['validation_datasets'] = dict(test=test_text)
if 'restore_path' not in test_kwargs:
    checkpoint_appendix = 'checkpoints/all_vars/best' if 'subgraphs_to_save' in train_kwargs else 'checkpoints/best'
    test_kwargs['restore_path'] = os.path.join(save_path, checkpoint_appendix)
test_kwargs['save_path'] = os.path.join(save_path, 'testing')


def train(q, idx):
    if config['seed'] is not None:
        tf.set_random_seed(config['seed'])
    test_kwargs['save_path'] += '/' + str(idx)
    train_kwargs['save_path'] += '/' + str(idx)
    env.build_pupil(**kwargs_for_building)
    env.train(**train_kwargs)
    _, _, mean_metrics = env.test(**test_kwargs)
    q.put(mean_metrics)

metrics = list()
for idx in range(config['num_repeats']):
    print("LAUNCH NUMBER %s" % idx)
    q = mp.Queue()
    p = mp.Process(target=train, args=(q, idx))
    p.start()
    metrics.append(q.get())
    p.join()

keys = list(metrics[0]['test'].keys())
metrics_ = {key: list() for key in keys}
# print('(run_train_test)metrics:', metrics)
for launch_res in metrics:
    for key in keys:
        # print('(run_train_test)launch_res:', launch_res)
        # print('(run_train_test)key:', key)
        # print('(run_train_test)metrics_:', metrics_)
        metrics_[key].append(launch_res['test'][key])

with open(os.path.join(save_path, 'test_mean_and_stddev.txt'), 'w') as f:
    for key, metric_values in metrics_.items():
        f.write("%s %s %s\n" % (key, sum(metric_values) / len(metric_values), compute_stddev(metric_values)))
