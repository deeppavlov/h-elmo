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
    os.path.join('/cephfs', os.path.expanduser('~/repos/learning-to-learn')),
    os.path.expanduser('~/repos/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/repos/h-elmo')),
    os.path.expanduser('~/repos/h-elmo'),
    '/cephfs/home/peganov/learning-to-learn',
    '/home/peganov/learning-to-learn',
    '/cephfs/home/peganov/h-elmo',
    '/home/peganov/h-elmo',
]


from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary, \
    compose_hp_confs, get_num_exps_and_res_files, compute_stddev

from helmo.util import organise

config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)

exec(organise.form_load_cmd(config['batch_gen']['path'], config['batch_gen']['cls_name'], "BatchGenerator"))
exec(organise.form_load_cmd(config['net']['path'], config['net']['cls_name'], "Net"))

save_path_relative_to_expres = '.'.join(config_path.split('.')[:-1])
# print(save_path_relative_to_expres)
results_dir = os.path.join(
    organise.append_path_after_experiments_to_expres_rm_head(save_path_relative_to_expres),
    os.path.split(save_path_relative_to_expres)[-1]
)
# print(results_dir)
save_path = results_dir

metrics, launches_for_testing, trained_launches = organise.load_tt_results(save_path, config['test']['result_types'])
# print("(run_train_test)metrics:", metrics)
# print("(run_train_test)launches_for_testing:", launches_for_testing)
# print("(run_train_test)trained_launches:", trained_launches)


dconf = config['dataset']
text = organise.get_text(dconf['path'])
test_size = int(dconf['test_size'])
valid_size = int(dconf['valid_size'])
train_size = int(dconf['train_size']) if 'train_size' in dconf else len(text) - test_size - valid_size
test_text, valid_text, train_text = organise.split_text(text, test_size, valid_size, train_size)

vocabulary, vocabulary_size = organise.get_vocab(dconf['vocab_path'], text)

env = Environment(Net, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

kwargs_for_building = config["graph"]
kwargs_for_building['voc_size'] = vocabulary_size

train_kwargs = config['train']
train_kwargs['train_dataset_text'] = train_text
train_kwargs['vocabulary'] = vocabulary
train_kwargs['validation_datasets'] = dict(valid=valid_text)

test_kwargs = config['test']
test_kwargs['vocabulary'] = vocabulary
test_kwargs['validation_datasets'] = dict(test=test_text)
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


for launch_folder in sorted(set(map(str, range(config['num_repeats']))) - set(trained_launches)):
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
