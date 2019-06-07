import string
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


def reduce_last_dim_v2(inp_tensor, target_last_dim):
    with tf.name_scope('reduce_last_dim_v2'):
        N = inp_tensor.get_shape().as_list()[-1]
        squeezing_matrix = squeezing_dense_matrix_tf(N, target_last_dim)
        return tf.einsum('ijk,kl->ijl', inp_tensor, squeezing_matrix)


def tile_to_match_target(inp_tensor, target_tensor):
    with tf.name_scope('tile_to_match_target'):
        num_dims = tf.shape(tf.shape(inp_tensor))
        num_repeats = tf.shape(target_tensor) // tf.shape(inp_tensor)
        repeated = tf.tile(inp_tensor, num_repeats + 1)
        return tf.slice(repeated, tf.zeros(num_dims, dtype=tf.int32), tf.shape(target_tensor))


def adjust_last_dim(inp_tensor, target_dim):
    with tf.name_scope('adjust_last_dim'):
        num_dims = tf.shape(tf.shape(inp_tensor))
        target_shape = tf.concat([tf.shape(inp_tensor)[:-1], tf.reshape(target_dim, [1])], 0)
        num_repeats = target_shape // tf.shape(inp_tensor)
        repeated = tf.tile(inp_tensor, num_repeats + 1)
        return tf.slice(repeated, tf.zeros(num_dims, dtype=tf.int32), target_shape)


def get_shapes(nested):
    if type(nested) == list:
        res = [get_shapes(e) for e in nested]
    elif type(nested) == tuple:
        res = tuple(get_shapes(e) for e in nested)
    elif type(nested) == dict:
        res = {k: get_shapes(v) for k, v in nested.items()}
    elif type(nested) == collections.OrderedDict:
        res = collections.OrderedDict([(k, get_shapes(v)) for k, v in nested.items()])
    elif python.is_namedtuple_instance(nested):
        tuple_type = type(nested)
        res = tuple_type(**{k: get_shapes(v) for k, v in nested._asdict()})
    else:
        res = nested.get_shape()
    return res


def sample_tensor_slices(tensor, n, axis):
    if isinstance(n, int):
        n = tf.constant(n, dtype=tf.int32)
    indices = tf.slice(tf.random_shuffle(tf.range(0, n, dtype=tf.int32)), [0], tf.reshape(n, [1]))
    return tf.gather(tensor, indices, axis=axis)


def average_k(tensor, k, axis, random_sampling=False):
    with tf.name_scope('average_k'):
        tensor_shape = tf.shape(tensor)
        dim = tf.shape(tensor)[axis]
        quotient = dim // k
        num_sampled = k * quotient
        num_dims = len(tensor.get_shape().as_list())
        if random_sampling:
            for_averaging = sample_tensor_slices(tensor, num_sampled, axis)
        else:
            for_averaging = tf.slice(
                tensor, [0]*num_dims,
                tf.concat(
                    [tensor_shape[:axis], tf.reshape(num_sampled, [1]), tensor_shape[axis+1:]],
                    0
                )
            )
        sh = tf.shape(for_averaging)
        k = tf.constant([k]) if isinstance(k, int) else tf.reshape(k, shape=[1])
        new_shape = tf.concat(
            [
                sh[:axis],
                tf.reshape(quotient, [1]),
                k,
                sh[axis+1:]
            ],
            0
        )
        for_averaging = tf.reshape(for_averaging, shape=new_shape)
        return tf.reduce_mean(for_averaging, axis=axis+1, keepdims=False)


def self_outer_product_eq(num_dims, axis):
    letters = string.ascii_lowercase
    base_letters = letters[:-2]
    base = base_letters[:num_dims]
    f1 = base[:axis] + letters[-2] + base[axis+1:]
    f2 = base[:axis] + letters[-1] + base[axis+1:]
    prod = base[:axis] + letters[-2:] + base[axis+1:]
    return f1 + ',' + f2 + '->' + prod


def self_outer_product(tensor, axis):
    with tf.name_scope('self_outer_product'):
        num_dims = len(tensor.get_shape().as_list())
        eq = self_outer_product_eq(num_dims, axis)
        return tf.einsum(eq, tensor, tensor)


def outer_product(tensor1, tensor2, axis):
    with tf.name_scope('outer_product'):
        num_dims = len(tensor1.get_shape().as_list())
        eq = self_outer_product_eq(num_dims, axis)
        return tf.einsum(eq, tensor1, tensor2)


def covariance(tensor, reduced_axes, cov_axis):
    with tf.name_scope('covariance'):
        mean = tf.reduce_mean(tensor, axis=reduced_axes, keepdims=True)
        devs = tensor - mean
        dev_prods = self_outer_product(devs, cov_axis)
        return tf.reduce_mean(dev_prods, axis=reduced_axes)


def correlation(tensor, reduced_axes, cor_axis, epsilon=1e-12):
    with tf.name_scope('correlation'):
        nd = len(tensor.get_shape().as_list())
        if not tf.contrib.framework.is_tensor(cor_axis):
            cor_axis %= nd
        if not tf.contrib.framework.is_tensor(reduced_axes):
            reduced_axes = [a % nd for a in reduced_axes]
        tensor = tf.cast(tensor, tf.float32)
        cov = covariance(tensor, reduced_axes, cor_axis)
        _, variance = tf.nn.moments(tensor, axes=reduced_axes, keep_dims=True)
        var_cross_mul = self_outer_product(variance, cor_axis)
        var_cross_mul = tf.reduce_sum(var_cross_mul, axis=reduced_axes)
        return cov / tf.sqrt(var_cross_mul + tf.constant(epsilon))


def covariance_2t(tensor1, tensor2, reduced_axes, cov_axis):
    with tf.name_scope('covariance_2t'):
        mean1 = tf.reduce_mean(tensor1, axis=reduced_axes, keepdims=True)
        devs1 = tensor1 - mean1
        mean2 = tf.reduce_mean(tensor2, axis=reduced_axes, keepdims=True)
        devs2 = tensor2 - mean2
        dev_prods = outer_product(devs1, devs2, cov_axis)
        return tf.reduce_mean(dev_prods, axis=reduced_axes)


def correlation_2t(tensor1, tensor2, reduced_axes, cor_axis, epsilon=1e-12):
    with tf.name_scope('correlation_2t'):
        nd = len(tensor1.get_shape().as_list())
        if not tf.contrib.framework.is_tensor(cor_axis):
            cor_axis %= nd
        if not tf.contrib.framework.is_tensor(reduced_axes):
            reduced_axes = [a % nd for a in reduced_axes]
        tensor1 = tf.cast(tensor1, tf.float32)
        tensor2 = tf.cast(tensor2, tf.float32)
        cov = covariance_2t(tensor1, tensor2, reduced_axes, cor_axis)
        _, variance1 = tf.nn.moments(tensor1, axes=reduced_axes, keep_dims=True)
        _, variance2 = tf.nn.moments(tensor2, axes=reduced_axes, keep_dims=True)
        var_cross_mul = outer_product(variance1, variance2, cor_axis)
        var_cross_mul = tf.reduce_sum(var_cross_mul, axis=reduced_axes)
        return cov / tf.sqrt(var_cross_mul + tf.constant(epsilon))


def get_low_triangle_mask(n):
    with tf.name_scope('get_low_triangle_mask'):
        r = tf.range(n)
        row_indices = tf.tile(
            tf.reshape(r, tf.concat([tf.reshape(n, [1]), [1]], 0)),
            tf.concat([[1], tf.reshape(n, [1])], 0),
        )
        col_indices = tf.tile(
            tf.reshape(r, tf.concat([[1], tf.reshape(n, [1])], 0)),
            tf.concat([tf.reshape(n, [1]), [1]], 0),
        )
        return col_indices < row_indices


def get_all_values_except_specified(tensor, excluded):
    with tf.name_scope('get_all_values_except_specified'):
        tensor = tf.reshape(tensor, [-1])
        excluded = tf.reshape(excluded, [-1])
        excluded_shape = tf.shape(excluded)
        tensor_expanded = tf.reshape(tensor, [-1, 1])
        multiples = tf.concat([[1], excluded_shape], 0)
        tensor_expanded = tf.tile(tensor_expanded, multiples)
        masks = tf.cast(tf.equal(tf.cast(tensor_expanded, tf.int32), tf.cast(excluded, tf.int32)), tf.int32)
        mask = tf.reduce_sum(masks, [1])
        mask = tf.cast(tf.cast(mask, dtype=tf.bool), dtype=tf.int32) - 1
        return tf.boolean_mask(tensor, mask)


def expand_multiple_dims(tensor, num_dims, axes):
    """
    Inserts dimensions of 1 in `tensor`. Old `tensor` dimensions are
    moved to permuted to positions specified in `axes`. Number of dimensions
    in result is equal to `num_dims`.
    ```python
    tensor = tf.constant([[1, 2], [3, 4]])
    expand_multiple_dims(tensor, 4, [1, 3])  # [[[[1, 2]],
                                             #   [[3, 4]]]]

    expand_multiple_dims(tensor, 4, [3, 1])  # [[[[1, 3]],
                                             #   [[2, 4]]]]
    ```
    :param tensor: a `Tensor`
    :param num_dims: a `Tensor` of shape `[]`
    :param axes: a 1D `Tensor`
    :return: a `num_dims` dimensional `Tensor` with same data as `tensor`
    """
    with tf.name_scope('expand_multiple_dims'):
        if not tf.contrib.framework.is_tensor(tensor):
            tensor = tf.constant(tensor)
        if not tf.contrib.framework.is_tensor(axes):
            axes = tf.constant(axes, dtype=tf.int32)
        sh = tf.shape(tensor, out_type=tf.int32)
        nd = tf.shape(sh, out_type=tf.int32)[0]
        with tf.device('/cpu:0'):
            assert_axes_smaller_than_num_dims = tf.assert_less(
                axes, num_dims, message='`axes` has to be less than `num_dims`')
            check_num_dims = tf.assert_greater_equal(
                num_dims, nd,
                message='`num_dims` has to be greater or equal to number of dimensions in `tensor`'
            )
            ass_axes_bigger_or_equal_than_num_dims = tf.assert_greater_equal(axes, -num_dims)

        axes %= num_dims

        ones_for_expansion = tf.ones(tf.reshape(num_dims - nd, [1]), dtype=tf.int32)
        shape_for_expansion = tf.concat([sh, ones_for_expansion], 0)

        tensor = tf.reshape(tensor, shape_for_expansion)

        updates = tf.range(0, num_dims, 1, dtype=tf.int32)
        remained_positions = get_all_values_except_specified(tf.range(num_dims, dtype=tf.int32), axes)
        indices = tf.concat([axes, remained_positions], 0)
        indices = tf.reshape(indices, [-1, 1])
        perm_shape = tf.reshape(num_dims, [1])
        perm = tf.scatter_nd(indices, updates, perm_shape)

        with tf.control_dependencies(
                [check_num_dims, assert_axes_smaller_than_num_dims, ass_axes_bigger_or_equal_than_num_dims]
        ):
            return tf.transpose(tensor, perm=perm)


def move_axes_to_front(tensor, axes):
    nd = tf.shape(tf.shape(tensor))[0]
    remained_axes = get_all_values_except_specified(tf.range(nd), axes)
    perm = tf.concat([axes, remained_axes], 0)
    return tf.transpose(tensor, perm=perm)


def get_low_triangle_values(tensor, axes):
    with tf.name_scope('get_low_triangle_values'):
        sh = tf.shape(tensor)
        with tf.device('/cpu:0'):
            assert_op = tf.assert_equal(
                sh[axes[0]],
                sh[axes[1]],
                message="Triangle can not be taken from not square matrix. "
                        "Make sure that `axes` dims in `tensor` have equal size."
            )
        with tf.control_dependencies([assert_op]):
            n = sh[axes[0]]
            ndims = tf.shape(sh)[0]
            axes %= ndims
            r = tf.range(n)
            rows = tf.reshape(r, [-1, 1])
            cols = tf.reshape(r, [1, -1])
            mask = tf.less(cols, rows)
            tensor = move_axes_to_front(tensor, axes)
            return tf.boolean_mask(tensor, mask)


def get_corr_matrix_axes(reduced_axes, cor_axis):
    with tf.name_scope('get_corr_matrix_axes'):
        num_preceding_reduced = tf.reduce_sum(tf.cast(tf.less(reduced_axes, cor_axis), tf.int32))
        return tf.stack([cor_axis - num_preceding_reduced, cor_axis - num_preceding_reduced + 1])


def get_correlation_values(tensor, reduced_axes, cor_axis):
    with tf.name_scope('get_correlation_values'):
        nd = len(tensor.get_shape().as_list())
        cor_axis %= nd
        reduced_axes = [a % nd for a in reduced_axes]
        corr = correlation(tensor, reduced_axes, cor_axis, epsilon=1e-12)
        matrix_axes = get_corr_matrix_axes(reduced_axes, cor_axis)
        return get_low_triangle_values(corr, matrix_axes)


def get_correlation_values_2t(tensor1, tensor2, reduced_axes, cor_axis):
    with tf.name_scope('get_correlation_values'):
        nd = len(tensor1.get_shape().as_list())
        cor_axis %= nd
        reduced_axes = [a % nd for a in reduced_axes]
        corr = correlation_2t(tensor1, tensor2, reduced_axes, cor_axis, epsilon=1e-12)
        return corr


def corcov_loss(
        tensor, reduced_axes, cor_axis,
        punish='correlation', reduction='sum', norm='sqr', epsilon=1e-12
):
    """
    Computes mean or sum of norm of correlation or covariation between elements of `tensor`
    with different indices along `cor_axis` dim. Each set of values of indices along
    dims not included in `cor_axis` and `reduced_axes` is considered an ensemble.
    This means that mean or sum of norm of correlation or covariation is computed for each set of indices
    that does not include `cor_axis` and `reduced_axes` and then averaged along `reduced_axes`.
    Diagonal values are excluded when mean correlation or covariation are computed.

    The norm is specified in `norm` and can be either `'sqr'` or `'abs'`. If `norm` is `'abs'`
    than mean or sum of of absolute values of correlation or covariation is computed. Else mean of
    square of correlation or covariation is computed.

    Let tensor `tensor` have 3 dimensions, `cor_axis = 2` , `reduced_axes = [1]`,
     `reduction = 'mean'`, `norm = 'abs'`, and
    `tensor` shape be `[A, B, C]`. Than

    ```
    covarition_matrices = tf.einsum('ijk,ijl->ikl', tensor, tensor) / B
    cov_norm = tf.abs(covarition_matrices)
    mean_covariations = (tf.reduce_sum(cov_norm, [-2, -1]) - tf.linalg.trace(cov_norm)) / \
                         (C * (C-1))
    covariation_loss = tf.reduce_mean(mean_covariations)
    ```
    Correlation loss is computed the same way, except for
    division of covariation by multiplication of variances of
    analyzed random values.

    :param tensor: tf.Tensor of at least 2 dimensions.
    :param reduced_axes: list of ints
    :param cor_axis: int
    :param punish: either `'correlation'` or `'covariation'`
        depending on what is computed correlation or covariation
    :param reduction: if `'sum'` than the sum of correlation
        or covariation is returned. If it is 'mean' average
        is computed.
    :param norm: `'sqr'` or `'abs'`
    :param epsilon: a small float for division by zero when computing correlation
    :return: tf.Tensor of shape []
    """
    with tf.name_scope('corcov_loss'):
        f = correlation if punish == 'correlation' else covariance
        tensor_nd = len(tensor.get_shape().as_list())
        cor_axis %= tensor_nd
        reduced_axes = [a % tensor_nd for a in reduced_axes]
        corcov = f(tensor, reduced_axes, cor_axis, epsilon=epsilon)
        nd = len(corcov.get_shape().as_list())
        perm = list(range(nd))
        num_prec_reduced = len(list(filter(lambda x: x < cor_axis, reduced_axes)))
        i1, i2 = cor_axis - num_prec_reduced, cor_axis - num_prec_reduced + 1
        if i1 == nd - 3:
            perm[nd - 3], perm[nd - 2], perm[nd - 1] = nd - 1, i1, i2
        elif i1 < nd - 3:
            perm[i1], perm[i2], perm[nd - 2], perm[nd - 1] = nd - 2, nd - 1, i1, i2
        corcov = tf.transpose(corcov, perm=perm)
        norm_func = lambda x: x**2 if norm == 'sqr' else tf.abs(x)
        norm_m = norm_func(corcov)
        frob_norm = tf.reduce_sum(norm_m, axis=[-2, -1])
        trace = tf.linalg.trace(norm_m)
        s = tf.reduce_sum(frob_norm - trace, keepdims=False)
        if reduction == 'sum':
            return 0.5 * s
        last_dim = tf.shape(norm_m)[-1]
        last_dim = tf.to_float(last_dim)
        return s / (tf.to_float(tf.reduce_prod(tf.shape(norm_m))) * (last_dim - 1.) / last_dim)


def get_axis_quarters(tensor):
    last_dim = tf.cast(tf.shape(tensor)[-1], dtype=tf.float32)
    exponents = tf.range(0., last_dim, 1., dtype=tf.float32)
    powers = tf.math.pow(2., exponents)
    binary_format = tf.cast(tensor > 0, tf.float32)
    linear_combination = powers * binary_format
    numbers = tf.reduce_sum(linear_combination, axis=-1)
    return tf.cast(numbers, tf.int32)
