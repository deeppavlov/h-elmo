import sys
import os

import tensorflow as tf
sys.path += [
    '/cephfs/home/peganov/learning-to-learn',
    '/home/peganov/learning-to-learn',
    '/cephfs/home/peganov/h-elmo',
    '/home/peganov/h-elmo',
]
# sys.path += [
#     os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
#     os.path.expanduser('~/learning-to-learn'),
#     os.path.join('/cephfs', os.path.expanduser('~/h-elmo')),
#     os.path.expanduser('~/h-elmo'),
# ]
from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary
from helmo.nets.reslstm import Lstm, LstmFastBatchGenerator as BatchGenerator
import helmo.util.organise as organise
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--train",
    help="If provided model is trained",
    action="store_true"
)
parser.add_argument(
    "--test",
    help="If provided model is tested",
    action="store_true"
)
parser.add_argument(
    "--inference",
    help="If provided model is dialog is started",
    action="store_true"
)
parser.add_argument(
    "--text_path",
    help="Path to file with text. Text is splitted in the following way: if test_size is not zero,"
         " first test_size lines are used for testing, than if valid_size is not zero lines from "
         "test_size to test_size+valid_size are for validation. What remains is for training.",
    default=None,
)
parser.add_argument(
    "voc_path",
    help="Path to vocabulary. If there is no vocabulary available the new vocabulary is created from text.",
)
parser.add_argument(
    "-cv",
    "--create_vocabulary",
    help="Create new vocabulary",
    action="store_true"
)
parser.add_argument(
    "--restore_path",
    help="Path to the checkpoint from which model is restored",
    default=None
)
parser.add_argument(
    "--save_path",
    help="Path to directory where training and inference results and model is saved",
    default='results'
)
parser.add_argument(
    "--test_size",
    help="Test dataset size in number of lines.",
    default=100000
)
parser.add_argument(
    "--valid_size",
    help="Validation dataset size in number of lines.",
    default=10000
)
args = parser.parse_args()

if args.train or args.test:
    with open(args.text_path, 'r') as f:
        text = f.read()
    train_size = len(text.split('\n')) - args.test_size - args.valid_size
    test_text, valid_text, train_text = organise.split_text(text, args.test_size, args.valid_size, train_size)
else:
    text = None

vocabulary, vocabulary_size = organise.get_vocab_by_given_path(args.voc_path, text, create=args.create_vocabulary)

env = Environment(Lstm, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

metrics = ['bpc', 'perplexity', 'accuracy']

tf.set_random_seed(1)

lstm_map = dict(
    module_name='char_enc_dec',
    num_nodes=[1500, 1500],
    input_idx=None,
    output_idx=None,
)
env.build_pupil(
    lstm_map=lstm_map,
    num_out_layers=1,
    num_out_nodes=[],
    voc_size=vocabulary_size,
    emb_size=256,
    init_parameter=3.,
    num_gpus=1,
    metrics=metrics,
    optimizer='adam',
    dropout_rate=0.1,
)

NUM_UNROLLINGS = 200
BATCH_SIZE = 32
if args.train:
    learning_rate = dict(
        type='adaptive_change',
        max_no_progress_points=1000,
        decay=.5,
        init=9e-4,
        path_to_target_metric_storage=('default_1', 'loss')
    )
    stop_specs = dict(
        type='while_progress',
        max_no_progress_points=10,
        changing_parameter_name='learning_rate',
        path_to_target_metric_storage=('default_1', 'loss')
    )
    # stop_specs = 2000
    env.train(
        allow_growth=True,
        save_path=args.save_path,
        restore_path=args.restore_path,
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        num_unrollings=NUM_UNROLLINGS,
        vocabulary=vocabulary,
        checkpoint_steps=None,
        subgraphs_to_save=dict(char_enc_dec='base'),
        result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
        printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
        stop=stop_specs,
        train_dataset_text=train_text,
        validation_dataset_texts=[valid_text],
        results_collect_interval=1000,
        no_validation=False,
        validation_batch_size=BATCH_SIZE,
        valid_batch_kwargs=dict(
            num_unrollings=NUM_UNROLLINGS,
        ),
    )
if args.train:
    restore_path = os.path.join(args.save_path, 'checkpoints/all_vars/best')
else:
    restore_path = args.restore_path
if args.test:
    env.test(
        restore_path=restore_path,
        save_path=args.save_path + '/testing',
        vocabulary=vocabulary,
        validation_dataset_texts=[valid_text],
        printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
        validation_batch_size=BATCH_SIZE,
        valid_batch_kwargs=dict(
            num_unrollings=NUM_UNROLLINGS,
        ),
    )

if args.inference:
    env.inference(
        restore_path=restore_path,
        log_path=args.save_path + '/dialog_logs.txt',
        vocabulary=vocabulary,
        character_positions_in_vocabulary=cpiv,
        batch_generator_class=BatchGenerator,
    )
