import argparse
import os
import json

from helmo.util import interpreter
interpreter.extend_python_path_for_project()

import helmo.util.dataset
from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary
from helmo.nets.resrnn import Rnn, LmFastBatchGenerator as BatchGenerator
from helmo.util.text import preprocessing, postprocessing

parser = argparse.ArgumentParser()

parser.add_argument(
    "--train",
    help="Train model",
    action="store_true"
)
parser.add_argument(
    "--test",
    help="Test model. Model is tested after training.",
    action="store_true"
)
parser.add_argument(
    "--file_dialog",
    help="Feed replicas from input_replica_file to model. Dialog is started after training and testing."
         " To load trained model provide --restore_path. Model answers are saved to file_dialog_answers_file.",
    action="store_true"
)
parser.add_argument(
    "--cli_dialog",
    help="Start cli dialog with model. Dialog is started after training, testing, and file dialog."
         " To load trained model provide --restore_path. To finish dialog type FINISH. Dialog is logged into"
         " cli_dialog_file",
    action="store_true"
)
parser.add_argument(
    "--telegram_bot",
    help="Start telegram bot. Telegram bot is started after training, testing, file dialog, and cli dialog."
         " To load trained model provide --restore_path. Dialogs are logged into directory specified in "
         "parameter 'telegram_log_path'. If parameter 'telegram_log_path' is not provided than dialogs are"
         " logged in directory 'telegram'.",
    action='store_true',
)
parser.add_argument(
    "--text_path",
    help="Path to file with text. Text is splitted in the following way: if test_size is not zero,"
         " first test_size lines are used for testing, than if valid_size is not zero lines from "
         "test_size to test_size+valid_size are for validation. What remains is for training.",
)
parser.add_argument(
    "--voc_path",
    help="Path to vocabulary. If there is no vocabulary available the new vocabulary is created from text.",
)
parser.add_argument(
    "--create_vocabulary",
    help="Create new vocabulary",
    action="store_true"
)
parser.add_argument(
    "--restore_path",
    help="Path to the checkpoint from which model is restored",
)
parser.add_argument(
    "--save_path",
    help="Path to directory where training and inference results and model is saved. Default is 'results'.",
    default='results',
)
parser.add_argument(
    "--cli_dialog_file",
    help="Name of a file for saving dialog logs. Default is dialog_logs.txt",
    default='dialogs.txt',
)
parser.add_argument(
    "--input_replica_file",
    help="Name of a file with replicas for inference. Default is 'replicas.txt'",
    default='replicas.txt'
)
parser.add_argument(
    "--file_dialog_answers_file",
    help="Name of a file with bot answers in file dialog regime. Default is answers.txt",
    default='answers.txt',
)
parser.add_argument(
    "--telegram_log_path",
    help="Path to directory where dialogs will be saved. Default is 'telegram'",
    default="telegram",
)
parser.add_argument(
    "--test_size",
    help="Test dataset size in number of lines. Default is 100000",
    type=int,
    default=100000,
)
parser.add_argument(
    "--valid_size",
    help="Validation dataset size in number of lines. Default is 10000",
    type=int,
    default=10000,
)
parser.add_argument(
    "--preprocessing",
    help="name of function in helmo/util/text/preprocessing.py applied to text"
         "before creating vocabulary and feeding text to model. Available options"
         " are: (1) 'hat_uppercase' replaces uppercase letter with corresponding"
         " lowercase letter preceded by hat '^'. Default is None."
)
parser.add_argument(
    "--postprocessing",
    help="name of function in helmo/util/text/postprocessing.py applied to bot replica"
         "before saving and printing. Available options are: (1) 'hat_uppercase'"
         " replaces hat '^' followed by lowercase letter with uppercase letter"
         " by hat '^'. Default is None."
)
parser.add_argument(
    "--reset_state_after_model_answer",
    help="Reset rnn hidden state after each model answer",
    action="store_true",
)
parser.add_argument(
    "--do_not_randomize_hidden_state",
    help="Reset rnn hidden state with zeroes. If is not set hidden states "
         "are initialized with random values",
    action='store_true',
)
parser.add_argument(
    "--randomize_state_stddev",
    help="If flag --do_not_randomize_hidden_state is not set, on inference reset_op"
         " initializes hidden states with values sampled via tf.truncated_normal. "
         "This argument sets standard deviation for distribution. Default is 0.5.",
    type=float,
    default=0.5,
)
parser.add_argument(
    "--temperature",
    help="Activates sampling from character prediction distribution if is not zero. "
         "Can be any not negative float. (1)Every value in distribution is raised to"
         " a power 1 / T, where T is temperature. (2)Distribution is normalised.",
    type=float,
    default=0.,
)
parser.add_argument(
    "--embed_inputs",
    help="Multiply input one hot vectors by matrix and bias.",
    action="store_true",
)
parser.add_argument(
    "--rnn_type",
    help="Type of rnn used in model. 'lstm' and 'gru' options are available."
         " Default is 'gru'",
    default='gru',
)
parser.add_argument(
    "--num_nodes",
    help="Number of nodes in recurrent neural network.",
    nargs='+',
    type=int,
    default=[1500, 1500]
)
parser.add_argument(
    "--optimizer",
    help="Optimizer for training. Default is 'adam'. 'sgd', 'adadelta', 'momentum',"
         " 'rmsprop', 'nesterov', 'adagrad' are available.",
    default='adam',
)
parser.add_argument(
    "--num_unrollings",
    help="Number of characters in sequence model processes in one run."
         " It is also depth of backpropagation through time.",
    type=int,
    default=200,
)
parser.add_argument(
    "--results_collect_interval",
    help="Number of steps done between logging",
    type=int,
    default=1000,
)
parser.add_argument(
    "--stop_specs",
    help="Specifies on which conditions training is stopped. If stop_specs is integer training is"
         " stopped after 'stop_specs' operations. Alternatively you can pass a dict "
         " through json config. Dictionary has to have item "
         "'max_no_progress_points'. If loss on validation dataset did not improve "
         "over stop_specs['max_no_progress_points'] last validations and learning_rate"
         " did not change or learning_rate changed twice while no progress were made"
         " during stop_specs['max_no_progress_points'] last validations "
         " after last time learning_rate has changed,"
         " training is stopped.",
    default='1000',
)
parser.add_argument(
    "--json_config",
    help="File with this script arguments. If provided values from file used as default but can be overridden by"
         " specifying arguments from command line. Attention: relative paths in json_config are provided relative"
         " to current working directory.",
)
args = parser.parse_args()

if args.stop_specs.isdigit():
    args.stop_specs = int(args.stop_specs)

config = vars(args)
if args.json_config is not None:
    with open(args.json_config, 'r') as f:
        json_config = json.load(f)
    for k, v in json_config.items():
        config[k] = v
    del config['json_config']
    
if config['preprocessing'] is not None:
    preprocess_f = getattr(preprocessing, config['preprocessing'])
else:
    preprocess_f = None
if config['postprocessing'] is not None:
    postprocess_f = getattr(postprocessing, config['postprocessing'])
else:
    postprocess_f = None

if config['train'] or config['test']:
    with open(os.path.expanduser(config['text_path']), 'r') as f:
        text = f.read()
    train_size = len(text.split('\n')) - config['test_size'] - config['valid_size']
    test_text, valid_text, train_text = helmo.util.dataset.split_text(
        text, config['test_size'], config['valid_size'], train_size, by_lines=True)
    if preprocess_f is not None:
        text = preprocess_f(text)
        valid_text = preprocess_f(valid_text)
        test_text = preprocess_f(test_text)
        train_text = preprocess_f(train_text)
else:
    text = None

# print("(dialog)len(text):", len(text))
# print("(dialog)len(valid_text):", len(valid_text))
# print("(dialog)len(train_text):", len(train_text))
# print("(dialog)len(test_text):", len(test_text))

vocabulary, vocabulary_size = helmo.util.dataset.get_vocab_by_given_path(
    os.path.expanduser(config['voc_path']), text, create=config['create_vocabulary'])

env = Environment(Rnn, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

metrics = ['bpc', 'perplexity', 'accuracy']

# tf.set_random_seed(1)

rnn_map = dict(
    module_name='char_enc_dec',
    num_nodes=args.num_nodes,
    input_idx=None,
    output_idx=None,
)
kwargs_for_model_building = dict(
    rnn_type=config['rnn_type'],
    embed_inputs=config['embed_inputs'],
    rnn_map=rnn_map,
    num_out_nodes=[],
    voc_size=vocabulary_size,
    emb_size=256,
    init_parameter=3.,
    num_gpus=1,
    metrics=metrics,
    optimizer=config['optimizer'],
    dropout_rate=0.1,
    randomize_state_stddev=config["randomize_state_stddev"]
)

env.build_pupil(
    **kwargs_for_model_building
)

BATCH_SIZE = 32
restore_path = None if config['restore_path'] is None else os.path.expanduser(config['restore_path'])
if config['train']:
    learning_rate = dict(
        type='adaptive_change',
        max_no_progress_points=10,
        decay=.5,
        init=4e-4,
        path_to_target_metric_storage=('valid', 'loss')
    )
    stop_specs = config['stop_specs']
    if isinstance(stop_specs, dict):
        stop_specs['changing_parameter_name'] = "learning_rate"
        stop_specs['path_to_target_metric_storage'] = ["valid", "loss"]
        stop_specs['type'] = "while_progress"
    env.train(
        allow_growth=True,
        save_path=os.path.expanduser(config['save_path']),
        restore_path=restore_path,
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        num_unrollings=config['num_unrollings'],
        vocabulary=vocabulary,
        checkpoint_steps=None,
        subgraphs_to_save=dict(char_enc_dec='base'),
        result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
        printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
        stop=stop_specs,
        train_dataset_text=train_text,
        validation_datasets={'valid': valid_text},
        results_collect_interval=config['results_collect_interval'],
        no_validation=False,
        validation_batch_size=BATCH_SIZE,
        valid_batch_kwargs=dict(
            num_unrollings=config['num_unrollings'],
        ),
        log_launch=False,
    )

if config['train']:
    restore_path = os.path.join(os.path.expanduser(config['save_path']), 'checkpoints/all_vars/best')
if config['test']:
    env.test(
        restore_path=restore_path,
        save_path=os.path.expanduser(config['save_path']) + '/testing',
        vocabulary=vocabulary,
        validation_datasets={'test': test_text},
        printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
        validation_batch_size=BATCH_SIZE,
        valid_batch_kwargs=dict(
            num_unrollings=config['num_unrollings'],
        ),
        log_launch=False,
    )
if config['file_dialog']:
    env.file_dialog(
        restore_path=restore_path,
        vocabulary=vocabulary,
        input_replica_file=os.path.expanduser(config['input_replica_file']),
        result_file=os.path.expanduser(config['file_dialog_answers_file']),
        character_positions_in_vocabulary=cpiv,
        batch_generator_class=BatchGenerator,
        reset_state_after_model_answer=args.reset_state_after_model_answer,  # if True after bot answer hidden state is reset
        answer_len_limit=500.,  # max number of characters in bot answer
        randomize=not config['do_not_randomize_hidden_state'],  # if True model hidden state is initialized with random numbers
        preprocess_f=preprocess_f,
        postprocess_f=postprocess_f,
        temperature=config['temperature'],
    )
if config['cli_dialog']:
    env.cli_dialog(
        restore_path=restore_path,
        log_file=os.path.expanduser(config['cli_dialog_file']),
        vocabulary=vocabulary,
        character_positions_in_vocabulary=cpiv,
        batch_generator_class=BatchGenerator,
        reset_state_after_model_answer=args.reset_state_after_model_answer,  # if True after bot answer hidden state is reset
        append_logs=False,  # if False and log_file already exists logs are put in to log_file + `#[index]`
        answer_len_limit=500.,  # max number of characters in bot answer
        randomize=not config['do_not_randomize_hidden_state'],  # if True model hidden state is initialized with random numbers
        preprocess_f=preprocess_f,
        postprocess_f=postprocess_f,
        temperature=config['temperature'],
    )
# print('before telegram method')
if config['telegram_bot']:
    env.telegram(
        kwargs_for_building=kwargs_for_model_building,
        restore_path=restore_path,
        vocabulary=vocabulary,
        character_positions_in_vocabulary=cpiv,
        batch_generator_class=BatchGenerator,
        log_path=config['telegram_log_path'],
        build=False,
        temperature=config['temperature'],
    )
