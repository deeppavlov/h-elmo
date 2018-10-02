import sys
import os
sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn')
]
import tensorflow as tf

from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary

from helmo.nets.reslstm import Lstm, LstmFastBatchGenerator as BatchGenerator

with open('../../datasets/razvedopros.txt', 'r') as f:
    text = f.read()

test_size = 2000
valid_size = 8000
train_size = 1000000
test_text = text[:test_size]
valid_text = text[test_size:test_size + valid_size]
train_text = text[test_size + valid_size:test_size + valid_size + train_size]
# valid_text = text[:valid_size]
# train_text = text[valid_size:]

voc_file_name = 'results/reslstm/razvedopros_voc.txt'
if os.path.isfile(voc_file_name):
    with open(voc_file_name, 'r') as f:
        vocabulary = list(f.read())
else:
    vocabulary = create_vocabulary(text)
    if not os.path.exists(voc_file_name):
        os.makedirs(os.path.dirname(voc_file_name))
    with open(voc_file_name, 'w') as f:
        f.write(''.join(vocabulary))
vocabulary_size = len(vocabulary)

env = Environment(Lstm, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

metrics = ['bpc', 'perplexity', 'accuracy']

# tf.set_random_seed(1)
NUM_UNROLLINGS = 30
BATCH_SIZE = 32

env.build_pupil(
    num_layers=2,
    num_nodes=[250, 250],
    num_out_layers=1,
    num_out_nodes=[],
    voc_size=vocabulary_size,
    emb_size=150,
    init_parameter=3.,
    num_gpus=1,
    metrics=metrics,
    optimizer='adam',
    dropout_rate=0.1,
)

print('building is finished')
stop_specs = dict(
    type='while_progress',
    max_no_progress_points=10,
    changing_parameter_name='learning_rate',
    path_to_target_metric_storage=('default_1', 'loss')
)
learning_rate = dict(
    type='adaptive_change',
    max_no_progress_points=1000,
    decay=.5,
    init=1e-3,
    path_to_target_metric_storage=('default_1', 'loss')
)
env.train(
    # gpu_memory=.3,
    allow_growth=True,
    save_path='lstm/start',
    restore_path='lstm/start/checkpoints/best',
    learning_rate=learning_rate,
    batch_size=BATCH_SIZE,
    num_unrollings=NUM_UNROLLINGS,
    vocabulary=vocabulary,
    checkpoint_steps=100,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    # stop=stop_specs,
    stop=40000,
    train_dataset_text=train_text,
    # train_dataset_text='abc',
    validation_dataset_texts=[valid_text],
    results_collect_interval=100,
    no_validation=False
)
# log_device_placement=True)
