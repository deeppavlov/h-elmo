import os
import copy
import tensorflow as tf
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import MaxBytesInUse
from tensorflow.contrib.cudnn_rnn import CudnnLSTM as CudnnLSTM
# from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import CudnnLSTM as CudnnLSTM
from tensorflow.contrib.rnn import LSTMBlockFusedCell as LSTMFused
from tensorflow.nn.rnn_cell import LSTMCell as LSTMCell
import time
import json
import argparse
import multiprocessing as mp
from useful_functions import all_combs, check_header_line_is_present, write_line, create_even_distribution


EXPERIMENT_PARAMS_ORDER = \
    ['sequence_length', 'batch_size', 'num_layers', 'num_units', 'num_parallel_launches', 'num_steps']


parser = argparse.ArgumentParser()
parser.add_argument(
    'path_to_config',
    help="Path to config of experiment",
)
parser.add_argument(
    '--machine',
    help="Name of machine on which computations are performed. If provided used as first directory in path to saved"
         "results",
    default=None,
)
args = parser.parse_args()


class UnknownConfigContent(Exception):
    def __init__(self, field, value, message):
        self.field = field
        self.value = value
        self.message = message


def create_inps_and_lbls(sequence_length, batch_size, num_units):
    inps = tf.ones([sequence_length, batch_size, num_units])
    lbls = tf.one_hot(tf.zeros([sequence_length, batch_size], dtype=tf.int32), num_units)
    return inps, lbls


def build_stacked_cudnn_lstm(inps, num_layers, num_units):
    lstms = [CudnnLSTM(1, num_units, input_mode='linear_input', ) for _ in range(num_layers)]
    inter = inps
    for lstm in lstms:
        inter, _ = lstm(inter)
    return inter


def build_cudnn_lstm(inps, num_layers, num_units):
    lstm = CudnnLSTM(num_layers, num_units, input_mode='linear_input', )
    output, _ = lstm(inps)
    return output


# def build_stacked_cudnn_lstm_graph(inps, num_layers, batch_size, num_units):
#     # lstms = [CudnnLSTM(1, num_units, num_units, input_mode='skip_input',) for _ in range(num_layers)]
#     lstms = [CudnnLSTM(1, num_units, input_mode='skip_input', ) for _ in range(num_layers)]
#     inter = inps
#     h_c = tf.zeros([batch_size, num_units])
#     for lstm in lstms:
#         inter, _ = lstm(inter)
#     return inter


def build_lstm_cell(inps, num_layers, num_units):
    lstms = [LSTMCell(num_units, dtype=tf.float32) for _ in range(num_layers)]
    multilayer_lstm = tf.contrib.rnn.MultiRNNCell(lstms)
    zero_state = multilayer_lstm.zero_state(tf.shape(inps)[1], tf.float32)
    logits, _ = tf.nn.dynamic_rnn(
        multilayer_lstm, inps, initial_state=zero_state, parallel_iterations=1024, time_major=True
    )
    return logits


def build_lstm_fused(inps, num_layers, num_units):
    lstms = [LSTMFused(num_units,) for _ in range(num_layers)]
    inter = inps
    for lstm in lstms:
        inter, _ = lstm(inter, dtype=tf.float32)
    return inter


def create_train_op(logits, lbls):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,
            labels=lbls,
        )
    )
    opt = tf.train.AdamOptimizer(0.)
    train_op = opt.minimize(loss)
    return train_op


def build_graph(config):
    inps, lbls = create_inps_and_lbls(config['sequence_length'], config['batch_size'], config['num_units'])
    if config['lstm_type'] == 'cudnn_stacked':
        logits = build_stacked_cudnn_lstm(inps, config['num_layers'], config['num_units'])
    elif config['lstm_type'] == 'fused':
        logits = build_lstm_fused(inps, config['num_layers'], config['num_units'])
    elif config['lstm_type'] == 'cudnn':
        logits = build_cudnn_lstm(inps, config['num_layers'], config['num_units'])
    elif config['lstm_type'] == 'cell':
        logits = build_lstm_cell(inps, config['num_layers'], config['num_units'])
    else:
        logits = None
    if config['mode'] == 'train':
        train_op = create_train_op(logits, lbls)
    else:
        train_op = None
    return train_op, logits


def measure_op_time(num_steps, op):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        start = time.time()
        for step in range(num_steps):
            if step > 0 and step % 100 == 0:
                print('%s iterations done' % step)
            sess.run(op)
        end = time.time()
    return (end - start) / num_steps


def measure_allocated_memory(num_steps, op):
    bytes_in_use = MaxBytesInUse()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(num_steps):
            sess.run(op)
        bytes_in_use = sess.run(bytes_in_use)
    return bytes_in_use


def perform_one_mesurement(config):
    train_op, logits = build_graph(config)
    if config['mode'] == 'train':
        op = train_op
    elif config['mode'] == 'infer':
        op = logits
    else:
        raise UnknownConfigContent(
            'mode',
            config['mode'],
            "Unknown config content.\n Field: '%s', value: %s" % ('mode', config['mode'])
        )
    if config['measured_spec'] == 'time':
        spec = measure_op_time(config['num_steps'], op)
    elif config['measured_spec'] == 'memory':
        spec = measure_allocated_memory(config['num_steps'], op)
    else:
        spec = None
    return spec


def make_list(candidate):
    if isinstance(candidate, list):
        l = candidate
    else:
        l = [candidate]
    return l


def split_experiment_config_into_separate_measurement_configs(config):
    seq_lens = make_list(config['sequence_length'])
    batch_sizes = make_list(config['batch_size'])
    num_layers = make_list(config['num_layers'])
    num_units = make_list(config['num_units'])
    combs = all_combs([seq_lens, batch_sizes, num_layers, num_units])
    configs = list()
    for comb in combs:
        conf = copy.deepcopy(config)
        conf['sequence_length'] = comb[0]
        conf['batch_size'] = comb[1]
        conf['num_layers'] = comb[2]
        conf['num_units'] = comb[3]
        configs.append(conf)
    if 'num_repeats' not in config or config['num_repeats'] == 1:
        return configs
    else:
        confs = list()
        for conf in configs:
            confs.extend([conf]*config['num_repeats'])
        return confs


def verify_header(configs, save_path):
    if configs[0]['measured_spec'] == 'time':
        res_name = 'op_time'
    elif configs[0]['measured_spec'] == 'memory':
        res_name = 'op_memory'
    else:
        raise UnknownConfigContent(
            'measured_spec',
            config['measured_spec'],
            "Unknown value in field 'measured_spec' '%s'" % config['measured_spec'],
        )
    if not check_header_line_is_present(save_path):
        write_line([res_name], [p for p in EXPERIMENT_PARAMS_ORDER if p in configs[0]], save_path, clean=True)


def save_results(configs, res, save_path):
    verify_header(configs, save_path)
    for m, config in zip(res, configs):
        params = [config[p] for p in EXPERIMENT_PARAMS_ORDER if p in config]
        # params = [config[p] for p in EXPERIMENT_PARAMS_ORDER]
        write_line([m], params, save_path, clean=False)


def vanilla_consumption(config):
    # in MB for sequence_length=100, batch_size=128, num_layers=2, num_units=2000
    # working for all lstm variants
    base_consumption = 4500
    consumption = base_consumption * \
        (config['num_units'] / 2000)**2 * \
        (config['num_layers'] / 2) * \
        (config['batch_size'] / 128) * \
        (config['sequence_length'] / 100)
    return consumption


def approx_mem_consumption(config):
    # in MB for sequence_length=100, batch_size=128, num_layers=2, num_units=2000
    # working for all lstm variants
    if 'memory_table' not in config or config['memory_table'] is None:
        consumption = vanilla_consumption(config)
    else:
        pass
    return consumption


def num_consequent_repeats(list_, idx):
    current = list_[idx]
    i = 1
    while idx + i < len(list_) and list_[idx+i] == current:
        i += 1
    return i


def ceil(a):
    if int(a) != a:
        return int(a) + 1
    return int(a)


def get_configs_run_in_parallel(configs, counter):
    current_config = configs[counter]
    max_num_in_parallel = current_config['memory_per_experiment'] / approx_mem_consumption(current_config)
    num_in_parallel = min(max_num_in_parallel, current_config['max_num_parallel_processes'])
    num_repeats = num_consequent_repeats(configs, counter)
    configs_to_process = configs[counter:counter+num_repeats]
    num_launches_required = ceil(num_repeats / num_in_parallel)
    distribution = create_even_distribution(num_launches_required, num_repeats)
    list_of_launches = list()
    pointer = 0
    for num_proc in distribution:
        added_confs = copy.deepcopy(configs_to_process[pointer:pointer+num_proc])
        for conf in added_confs:
            conf['num_parallel_launches'] = num_proc
        list_of_launches.append(added_confs)
        pointer += num_proc
    return list_of_launches, num_repeats


with open(args.path_to_config, 'r') as f:
    config = json.load(f)
configs = split_experiment_config_into_separate_measurement_configs(config)
# build_graph(configs[0])
# saver = tf.train.Saver(saveable)
#
# weights_shapes = [[tf.shape(ww) for ww in w] for w in weights]
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(weights_shapes))
#     print(sess.run(sizes))
#     writer = tf.summary.FileWriter('results/linear')
#     graph = tf.get_default_graph()
#     print(graph)
#     writer.add_graph(graph)


if args.machine is not None and len(args.machine) > 0:
    dirs = config['save_path'].split('/')
    if 'results' in dirs:
        dirs.insert(dirs.index('results') + 1, args.machine)
    else:
        dirs = [args.machine] + dirs
    save_path = os.path.join(*dirs)
else:
    save_path = config['save_path']


num_measurements = len(configs)
counter = 0
if config['measured_spec'] == 'time':
    while counter < num_measurements:
        list_of_launches, num_processed = get_configs_run_in_parallel(configs, counter)
        for confs_to_run in list_of_launches:
            with mp.Pool(len(confs_to_run)) as p:
                res = p.map(perform_one_mesurement, confs_to_run)
            print(res)
            save_results(confs_to_run, res, save_path)
        counter += num_processed
elif config['measured_spec'] == 'memory':
    for i in range(num_measurements):
        config_to_run = [configs[i]]
        with mp.Pool(len(config_to_run)) as p:
            res = p.map(perform_one_mesurement, config_to_run)
        print(res)
        save_results(config_to_run, res, save_path)
else:
    raise UnknownConfigContent(
        'measured_spec',
        config['measured_spec'],
        "Unknown value in field 'measured_spec' '%s'" % config['measured_spec'],
    )


