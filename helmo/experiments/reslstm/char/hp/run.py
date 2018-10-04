import sys
import os
sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn'),
    os.path.join('/cephfs', os.path.expanduser('~/h-elmo')),
    os.path.expanduser('~/h-elmo'),
]
import tensorflow as tf

from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary, \
    compose_hp_confs, get_num_exps_and_res_files

from helmo.nets.reslstm import Lstm, LstmFastBatchGenerator as BatchGenerator
import helmo.util.organise as organise

parameter_set_file_name = sys.argv[1]
if len(sys.argv) > 2:
    chop_last_experiment = bool(sys.argv[2])
else:
    chop_last_experiment = False
conf_name = os.path.join(*parameter_set_file_name.split('.')[:-1])
rel_path = os.path.join(*os.path.split(os.path.abspath(__file__).split('/experiments/')[-1])[:-1])
dataset_name = 'valid'
save_path = os.path.join(organise.path_rel_to_root('expres'), rel_path, conf_name)
results_file_name = os.path.join(save_path, dataset_name + '.txt')
confs, _ = compose_hp_confs(
    parameter_set_file_name, results_file_name, chop_last_experiment=chop_last_experiment, model='pupil')
confs.reverse()  # start with small configs
print("confs:", confs)

dataset_file_name = 'enwiki1G.txt'
text = organise.get_text(dataset_file_name)

test_size, valid_size = int(6.4e6), int(6.4e5)
train_size = len(text) - test_size - valid_size
test_text, valid_text, train_text = organise.split_text(text, test_size, valid_size, train_size)

voc_file_name = 'enwiki1G_voc.txt'
vocabulary, vocabulary_size = organise.get_vocab(voc_file_name, text)


env = Environment(Lstm, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

metrics = ['bpc', 'perplexity', 'accuracy']

# tf.set_random_seed(1)

stop_specs = dict(
    type='while_progress',
    max_no_progress_points=10,
    changing_parameter_name='learning_rate',
    path_to_target_metric_storage=('valid', 'loss')
)

NUM_UNROLLINGS = 200
BATCH_SIZE = 32

evaluation = dict(
    save_path=save_path,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    datasets=[(valid_text, dataset_name)],
    batch_gen_class=BatchGenerator,
    batch_kwargs={'vocabulary': vocabulary, 'num_unrollings': 200},
    batch_size=BATCH_SIZE,
    additional_feed_dict=[],
)

lstm_map = dict(
    module_name='char_enc_dec',
    num_nodes=[1500, 1500],
    input_idx=None,
    output_idx=None,
    # derived_branches=[
    #     dict(
    #         module_name='word_enc_dec',
    #         num_nodes=[3000, 3000],
    #         input_idx=0,
    #         output_idx=1,
    #     )
    # ]
)
kwargs_for_building = dict(
    lstm_map=lstm_map,
    num_out_nodes=[],
    voc_size=vocabulary_size,
    emb_size=256,
    init_parameter=3.,
    num_gpus=1,
    metrics=metrics,
    optimizer='adam',
    dropout_rate=0.1,
    clip_norm=5.,
)

launch_kwargs = dict(
    allow_growth=True,
    # restore_path=dict(
    #     char_enc_dec='results/reslstm/checkpoints/all_vars/best',
    # ),
    learning_rate=-1,
    batch_size=BATCH_SIZE,
    num_unrollings=NUM_UNROLLINGS,
    vocabulary=vocabulary,
    checkpoint_steps=None,
    stop=200,
    train_dataset_text=train_text,
    no_validation=True,
    printed_result_types=None,
)

for conf in confs:
    build_hyperparameters = dict(
        init_parameter=conf['init_parameter']
    )
    other_hyperparameters = dict(
        learning_rate=dict(
            varying=dict(
                value=conf['learning_rate/value']
            ),
            hp_type='built-in',
            type='fixed'
        )
    )

    # tf.set_random_seed(1)
    _, biggest_idx, _ = get_num_exps_and_res_files(save_path)
    if biggest_idx is None:
        initial_experiment_counter_value = 0
    else:
        initial_experiment_counter_value = biggest_idx + 1
    env.grid_search(
        evaluation,
        kwargs_for_building,
        build_hyperparameters=build_hyperparameters,
        other_hyperparameters=other_hyperparameters,
        initial_experiment_counter_value=initial_experiment_counter_value,
        **launch_kwargs
    )
