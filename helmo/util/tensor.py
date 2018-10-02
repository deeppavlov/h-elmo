import tensorflow as tf

from helmo.util.nested import synchronous_flatten, deep_zip, apply_func_on_depth


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


def cudnn_cell_zero_state(batch_size, cell):
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


def get_zero_state(inps, lstms, lstm_type):
    batch_size = tf.shape(inps)[1]
    if lstm_type == 'cell':
        zero_state = lstms.zero_state(batch_size, tf.float32)
    if lstm_type == 'cudnn':
        zero_state = cudnn_cell_zero_state(batch_size, lstms)
    if lstm_type == 'cudnn_stacked':
        zero_state = [cudnn_cell_zero_state(batch_size, lstm) for lstm in lstms]
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


def prepare_init_state(saved_state, inps, lstms, lstm_type):
    # inps = tf.Print(inps, [tf.shape(inps)], message="(prepare_init_state)tf.shape(inps):\n")
    # saved_state = list(saved_state)
    # for idx, s in enumerate(saved_state):
    #     saved_state[idx] = tf.Print(s, [tf.shape(s)], message="(prepare_init_state)saved_state[%s].shape:\n" % idx)
    # saved_state = tuple(saved_state)
    if lstm_type == 'cudnn' and isinstance(lstms, list):
        lstm_type = 'cudnn_stacked'
    if saved_state is None:
        return None
    if lstm_type == 'cudnn':
        depth = 1
    elif lstm_type == 'cudnn_stacked':
        depth = 2
    else:
        depth = 0
    zero_state = get_zero_state(inps, lstms, lstm_type)
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


def compute_lstm_stddevs(num_units, input_dim, init_parameter):
    stddevs = list()
    prev_nu = input_dim
    for nu in num_units:
        stddevs.append(init_parameter / (prev_nu + 2 * nu)**.5)
        prev_nu = nu
    return stddevs


def get_saved_state_vars(num_layers, lstm_type):
    if lstm_type == 'cudnn':
        state = (
            tf.Variable(0., trainable=False, validate_shape=False, name='lstm_h'),
            tf.Variable(0., trainable=False, validate_shape=False, name='lstm_c'),
        )
    elif lstm_type == 'cudnn_stacked':
        state = [
            (
                tf.Variable(0., trainable=False, validate_shape=False, name='lstm_%s_h' % idx),
                tf.Variable(0., trainable=False, validate_shape=False, name='lstm_%s_c' % idx),
            ) for idx in range(num_layers)
        ]
    elif lstm_type == 'cell':
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
    else:
        state = None
    # print(state)
    return state