import collections

import numpy as np
from scipy.sparse import coo_matrix
import tensorflow as tf

from helmo.util.nested import synchronous_flatten, deep_zip, apply_func_on_depth
import helmo.util.python as python


def compose_save_list(*pairs, name_scope='save_list'):
    with tf.name_scope(name_scope):
        save_list = list()
        for pair in pairs:
            # print('pair:', pair)
            [variables, new_values] = synchronous_flatten(pair[0], pair[1])
            # print("(useful_functions.compose_save_list)variables:", variables)
            # variables = flatten(pair[0])
            # # print(variables)
            # new_values = flatten(pair[1])
            for variable, value in zip(variables, new_values):
                save_list.append(tf.assign(variable, value, validate_shape=False))
        return save_list


def is_scalar_tensor(t):
    # print("(is_scalar_tensor)t:", t)
    return tf.equal(tf.shape(tf.shape(t))[0], 0)


def adjust_saved_state(saved_and_zero_state):
    # print("(replace_empty_saved_state)saved_and_zero_state:", saved_and_zero_state)

    # returned = tf.cond(
    #     is_scalar_tensor(saved_and_zero_state[0]),
    #     true_fn=lambda: saved_and_zero_state[1],
    #     false_fn=lambda: tf.reshape(
    #         saved_and_zero_state[0],
    #         tf.shape(saved_and_zero_state[1])
    #     ),
    # )
    name = saved_and_zero_state[0].name.split('/')[-1].replace(':', '_')
    zero_state_shape = tf.shape(saved_and_zero_state[1])
    # zero_state_shape = tf.Print(zero_state_shape, [zero_state_shape], message="(adjust_saved_state)zero_state_shape:\n")
    # zero_state_shape = tf.Print(
    #     zero_state_shape, [tf.shape(saved_and_zero_state[0])], message="(adjust_saved_state)saved_and_zero_state[0]:\n")
    returned = tf.cond(
        is_scalar_tensor(saved_and_zero_state[0]),
        true_fn=lambda: saved_and_zero_state[1],
        false_fn=lambda: adjust_batch_size(saved_and_zero_state[0], zero_state_shape[-2]),
    )
    # print("(replace_empty_saved_state)returned:", returned)
    return tf.reshape(returned, zero_state_shape, name=name + "_adjusted")


def cudnn_lstm_zero_state(batch_size, cell):
    zero_state = tuple(
        [
            tf.zeros(
                tf.concat(
                    [
                        [cell.num_dirs * cell.num_layers],
                        tf.reshape(batch_size, [1]),
                        [cell.num_units]
                    ],
                    0
                )
            )
        ] * 2
    )
    return zero_state


def cudnn_gru_zero_state(batch_size, cell):
    return (tf.zeros(
        tf.concat(
            [
                [cell.num_dirs * cell.num_layers],
                tf.reshape(batch_size, [1]),
                [cell.num_units]
            ],
            0
        )
    ), )


def get_zero_state(inps, rnns, rnn_type):
    batch_size = tf.shape(inps)[1]
    if rnn_type == 'cell':
        zero_state = rnns.zero_state(batch_size, tf.float32)
    elif rnn_type == 'cudnn_lstm':
        zero_state = cudnn_lstm_zero_state(batch_size, rnns)
    elif rnn_type == 'cudnn_lstm_stacked':
        zero_state = [cudnn_lstm_zero_state(batch_size, lstm) for lstm in rnns]
    elif rnn_type == 'cudnn_gru':
        zero_state = cudnn_gru_zero_state(batch_size, rnns)
    elif rnn_type == 'cudnn_gru_stacked':
        zero_state = [cudnn_gru_zero_state(batch_size, gru) for gru in rnns]
    # print("(get_zero_state)zero_state:", zero_state)
    return zero_state


def adjust_batch_size(state, batch_size):
    # batch_size = tf.Print(batch_size, [batch_size], message="(discard_redundant_states)batch_size:\n")
    # batch_size = tf.Print(batch_size, [tf.shape(state)], message="(discard_redundant_states)tf.shape(state):\n")
    state_shape = tf.shape(state)
    added_states_shape = tf.concat(
        [
            tf.shape(state)[:-2],
            tf.reshape(batch_size - state_shape[-2], [1]),
            tf.shape(state)[-1:]
        ],
        0
    )
    slice_start = tf.zeros(tf.shape(tf.shape(state)), dtype=tf.int32)
    remaining_states_shape = tf.concat(
        [
            tf.shape(state)[:-2],
            tf.reshape(batch_size, [1]),
            tf.shape(state)[-1:]
        ],
        0
    )
    return tf.cond(
        tf.shape(state)[-2] < batch_size,
        true_fn=lambda: tf.concat([state, tf.zeros(added_states_shape)], -2, name='extended_state'),
        false_fn=lambda: tf.slice(state, slice_start, remaining_states_shape, name='shortened_state'),
        name='state_with_adjusted_batch_size'
    )


def prepare_init_state(saved_state, inps, rnns, rnn_type):
    # inps = tf.Print(inps, [tf.shape(inps)], message="(prepare_init_state)tf.shape(inps):\n")
    # saved_state = list(saved_state)
    # for idx, s in enumerate(saved_state):
    #     saved_state[idx] = tf.Print(s, [tf.shape(s)], message="(prepare_init_state)saved_state[%s].shape:\n" % idx)
    # saved_state = tuple(saved_state)
    # print("(tensor.prepare_init_state)saved_state:", saved_state)
    # print("(tensor.prepare_init_state)rnn_type:", rnn_type)
    if rnn_type == 'cudnn_lstm' and isinstance(rnns, list):
        rnn_type = 'cudnn_lstm_stacked'

    if rnn_type == 'cudnn_gru' and isinstance(rnns, list):
        rnn_type = 'cudnn_gru_stacked'
    if saved_state is None:
        return None

    if rnn_type == 'cudnn_lstm':
        depth = 1
    elif rnn_type == 'cudnn_lstm_stacked' or rnn_type == 'cudnn_gru_stacked':
        depth = 2
    elif rnn_type == 'cudnn_gru':
        depth = 1
    else:
        depth = 0

    with tf.name_scope('prepare_init_state'):
        zero_state = get_zero_state(inps, rnns, rnn_type)
        zero_and_saved_states_zipped = deep_zip(
            [saved_state, zero_state], -1
        )
        # print("(prepare_init_state)zero_and_saved_states_zipped:", zero_and_saved_states_zipped)
        returned = apply_func_on_depth(zero_and_saved_states_zipped, adjust_saved_state, depth)
    # returned = apply_func_on_depth(
    #     returned,
    #     lambda x: discard_redundant_states(
    #         x,
    #         tf.shape(inps)[1],
    #     ),
    #     depth
    # )
    # print("(prepare_init_state)returned:", returned)
    return returned


def compute_lstm_gru_stddevs(num_units, input_dim, init_parameter):
    stddevs = list()
    prev_nu = input_dim
    for nu in num_units:
        stddevs.append(init_parameter / (prev_nu + 2 * nu)**.5)
        prev_nu = nu
    return stddevs


def get_saved_state_vars(num_layers, rnn_type, name_scope='lstm_states'):
    with tf.name_scope(name_scope):
        if rnn_type == 'cudnn_lstm':
            state = (
                tf.Variable(0., trainable=False, validate_shape=False, name='lstm_h'),
                tf.Variable(0., trainable=False, validate_shape=False, name='lstm_c'),
            )
        elif rnn_type == 'cudnn_lstm_stacked':
            state = [
                (
                    tf.Variable(0., trainable=False, validate_shape=False, name='lstm_%s_h' % idx),
                    tf.Variable(0., trainable=False, validate_shape=False, name='lstm_%s_c' % idx),
                ) for idx in range(num_layers)
            ]
        elif rnn_type == 'cell':
            # state = LSTMStateTuple(
            #     h=[
            #         tf.Variable(0., trainable=False, validate_shape=False, name='cell_h_%s' % i)
            #         for i in range(num_layers)
            #     ],
            #     c=[
            #         tf.Variable(0., trainable=False, validate_shape=False, name='cell_c_%s' % i)
            #         for i in range(num_layers)
            #     ],
            # )
            state = tf.Variable(0., trainable=False, validate_shape=False, name='cell_state')
        elif rnn_type == 'cudnn_gru':
            state = (tf.Variable(0., trainable=False, validate_shape=False, name='gru_c'), )
        elif rnn_type == 'cudnn_gru_stacked':
            state = [
                (tf.Variable(0., trainable=False, validate_shape=False, name='gru_%s_c' % idx), )
                for idx in range(num_layers)
            ]
        else:
            state = None
        # print(state)
    return state


def indices_and_weights_for_squeezing_of_1_neuron(i, N, M):
    weights = []
    indices = []
    q = N / M
    left, right = i * q, (i + 1) * q
    if N >= M:
        if int(left) == left:
            start_of_ones = int(left)
        else:
            weights.append(np.ceil(left) - left)
            indices.append(int(left))
            start_of_ones = int(left) + 1

        end_of_ones = int(right)
        weights += [1.] * (end_of_ones - start_of_ones)
        indices.extend(range(start_of_ones, end_of_ones))

        if int(right) != right:
            weights.append(right - int(right))
            indices.append(indices[-1] + 1)
    else:
        left_floor, right_floor = int(left), int(right)
        if left_floor == right_floor or right == right_floor:
            weights.append(right - left)
            indices.append(left_floor)
        else:
            weights += [right_floor - left, right - right_floor]
            indices += [left_floor, right_floor]

    factor = sum([w**2 for w in weights]) ** -0.5
    weights = [w * factor for w in weights]
    return indices, weights


def squeezing_sparse_matrix(N, M):
    init_indices, init_weights = [], []
    for i in range(M):
        # print("(util.tensor.squeezing_sparse_matrix)N, M:", N, M)
        indices, weights = indices_and_weights_for_squeezing_of_1_neuron(i, N, M)
        init_indices.extend(zip(indices, [i] * len(indices)))
        init_weights += weights
    return init_indices, init_weights


def squeezing_sparse_matrix_tf(N, M):
    indices, weights = squeezing_sparse_matrix(N, M)
    return tf.SparseTensor(indices, weights, [N, M])


def squeezing_dense_matrix_tf(N, M):
    indices, weights = squeezing_sparse_matrix(N, M)
    row_ids, col_ids = zip(*indices)
    sp_matrix = coo_matrix((weights, (row_ids, col_ids)), shape=(N, M))
    return tf.constant(sp_matrix.todense(), dtype=tf.float32)


def reduce_last_dim(inp_tensor, target_tensor):
    # print("(tensor.reduce_last_dim)inp_tensor:", inp_tensor)
    # print("(tensor.reduce_last_dim)target_tensor:", target_tensor)
    with tf.name_scope('reduce_last_dim'):
        N = inp_tensor.get_shape().as_list()[-1]
        M = target_tensor.get_shape().as_list()[-1]
        squeezing_matrix = squeezing_dense_matrix_tf(N, M)
        return tf.einsum('ijk,kl->ijl', inp_tensor, squeezing_matrix)


def increase_last_dim(inp_tensor, target_tensor):
    with tf.name_scope('increase_last_dim'):
        num_dims = tf.shape(tf.shape(inp_tensor))
        num_repeats = tf.shape(target_tensor) // tf.shape(inp_tensor)
        repeated = tf.tile(inp_tensor, num_repeats + 1)
        return tf.slice(repeated, tf.zeros(num_dims, dtype=tf.int32), tf.shape(target_tensor))


def get_shapes(nested):
    if type(nested) == list:
        res = [get_shapes(e) for e in nested]
    elif type(nested) == tuple:
        res = tuple(get_shapes(e) for e in nested)
    elif type(nested) == dict:
        res = {k: get_shapes(v) for k, v in nested.items()}
    elif type(nested) == collections.OrderedDict:
        res = collections.OrderedDict([(k, get_shapes(v)) for k, v in nested.items()])
    elif python.isnamedtupleinstance(nested):
        tuple_type = type(nested)
        res = tuple_type(**{k: get_shapes(v) for k, v in nested._asdict()})
    else:
        res = nested.get_shape()
    return res
