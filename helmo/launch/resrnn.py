import sys
import os
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
import tensorflow as tf

from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary

from helmo.nets.resrnn import Rnn, LmFastBatchGenerator as BatchGenerator
import helmo.util.organise as organise

dataset_file_name = 'enwiki1G.txt'
text = organise.get_text(dataset_file_name)

test_size, valid_size = int(6.4e6), int(6.4e5)
train_size = len(text) - test_size - valid_size
test_text, valid_text, train_text = organise.split_text(text, test_size, valid_size, train_size)


voc_file_name = 'enwiki1G_voc.txt'
vocabulary, vocabulary_size = organise.get_vocab(voc_file_name, text)

env = Environment(Rnn, BatchGenerator, vocabulary=vocabulary)

metrics = ['bpc', 'perplexity', 'accuracy']

# tf.set_random_seed(1)

NUM_UNROLLINGS = 200
BATCH_SIZE = 32

rnn_map = dict(
    module_name='char_enc_dec',
    num_nodes=[100, 200, 150],
    input_idx=None,
    output_idx=None,
    derived_branches=[
        dict(
            module_name='word_enc_dec',
            num_nodes=[320, 600],
            input_idx=0,
            output_idx=1,
        )
    ]
)
env.build_pupil(
    rnn_type='lstm',
    embed_inputs=True,
    rnn_map=rnn_map,
    num_out_nodes=[],
    voc_size=vocabulary_size,
    emb_size=256,
    init_parameter=3.,
    num_gpus=1,
    metrics=metrics,
    optimizer='adam',
    dropout_rate=0.,
    # regime='inference',
    # backward_connections=True,
    # matrix_dim_adjustment=True,
)
learning_rate = dict(
    type='adaptive_change',
    max_no_progress_points=1,
    decay=.5,
    init=4e-4,
    # init=1e-3,
    path_to_target_metric_storage=('default_1', 'loss')
)
stop_specs = {
      "type": "while_progress",
      "max_no_progress_points": 1,
      "changing_parameter_name": "learning_rate",
      "path_to_target_metric_storage": ["default_1", "loss"]
    }
valid_tensor_schedule = dict(
    valid_pickle_mean_tensors=dict(correlation=1),
    valid_pickle_all_tensors=dict(correlation=1),
)
env.train(
    # gpu_memory=.3,
    allow_growth=True,
    # save_path='results/resrnn',
    # # restore_path='results/resrnn/checkpoints/all_vars/best',
    # restore_path='results/resrnn/checkpoints/best',

    save_path='results/resrnn/correlation',
    # restore_path='results/resrnn/back/checkpoints/best',

    # restore_path=dict(
    #     char_enc_dec='results/resrnn/checkpoints/all_vars/best',
    # ),
    learning_rate=learning_rate,
    lr_restore_saver_name='saver',
    batch_size=BATCH_SIZE,
    num_unrollings=NUM_UNROLLINGS,
    vocabulary=vocabulary,
    checkpoint_steps=None,
    # subgraphs_to_save=dict(char_enc_dec='base'),
    # subgraphs_to_save=['char_enc_dec', 'word_enc_dec'],
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=stop_specs,
    # stop=40000,
    train_dataset_text=train_text,
    # train_dataset_text='abc',
    validation_dataset_texts=[valid_text],
    results_collect_interval=100,
    no_validation=False,
    validation_batch_size=32,
    valid_batch_kwargs=dict(
        num_unrollings=100,
    ),
    add_graph_to_summary=True,
    summary=True,
    state_reset_period=10,
    validation_tensor_schedule=valid_tensor_schedule,
)

# rnn_map = dict(
#     module_name='char_enc_dec',
#     num_nodes=[250, 125],
#     input_idx=None,
#     output_idx=None,
#     derived_branches=[
#         dict(
#             module_name='word_enc_dec',
#             num_nodes=[500, 480],
#             input_idx=0,
#             output_idx=1,
#         )
#     ]
# )
#
# env.build_pupil(
#     rnn_map=rnn_map,
#     num_out_nodes=[],
#     voc_size=vocabulary_size,
#     emb_size=150,
#     init_parameter=3.,
#     num_gpus=1,
#     metrics=metrics,
#     optimizer='adam',
#     dropout_rate=0.1,
# )
# learning_rate = dict(
#     type='adaptive_change',
#     max_no_progress_points=1000,
#     decay=.5,
#     init=0.,
#     path_to_target_metric_storage=('default_1', 'loss')
# )
#
# env.train(
#     # gpu_memory=.3,
#     allow_growth=True,
#     save_path='results/resrnn',
#     restore_path=dict(
#         char_enc_dec='results/resrnn/checkpoints/all_vars/best',
#     ),
#     learning_rate=learning_rate,
#     batch_size=BATCH_SIZE,
#     num_unrollings=NUM_UNROLLINGS,
#     vocabulary=vocabulary,
#     checkpoint_steps=None,
#     subgraphs_to_save=dict(word_enc_dec='word_level', all_vars='full_model',),
#     # subgraphs_to_save=['char_enc_dec', 'word_enc_dec'],
#     result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#     printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#     # stop=stop_specs,
#     stop=40000,
#     train_dataset_text=train_text,
#     # train_dataset_text='abc',
#     validation_dataset_texts=[valid_text],
#     results_collect_interval=100,
#     no_validation=False,
#     validation_batch_size=32,
#     valid_batch_kwargs=dict(
#         num_unrollings=100,
#     ),
#     add_graph_to_summary=True,
# )
