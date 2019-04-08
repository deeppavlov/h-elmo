import numpy as np
import tensorflow as tf
import pytest

import helmo.util.tensor_ops as tensor_ops


gpu_options = tf.GPUOptions(allow_growth=True)


class TestGetLowTriangleValues:

    def test_dim_sizes_not_equal(self):
        axes = tf.constant([0, 1])
        t = tf.constant([[1, 2, 3], [4, 5, 6]])
        low_triangle = tensor_ops.get_low_triangle_values(t, axes)
        with tf.Session() as sess:
            with pytest.raises(tf.errors.InvalidArgumentError):
                sess.run(low_triangle)


class TestExpandMultipleDims:

    def test_case_no_changes_in_tensor(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            for init_num_dims in range(4):
                shape = list(range(1, 1 + init_num_dims))
                size = 1
                if len(shape) > 0:
                    for d in shape:
                        size *= d
                if init_num_dims == 0:
                    values = np.array(1.)
                else:
                    values = np.arange(0, size, 1)
                for _ in range(init_num_dims + 1):
                    shape = shape[-1:] + shape[:-1]
                    tens = tf.reshape(values, shape)
                    np_tens = np.reshape(values, shape)

                    axes = tf.range(init_num_dims)
                    num_dims = tf.constant(init_num_dims)

                    expanded = tensor_ops.expand_multiple_dims(tens, num_dims, axes)
                    msg_values = "tensor == {}\n" \
                        "num_dims == {}\n" \
                        "axes == {}".format(np_tens, init_num_dims, list(range(init_num_dims)))
                    assert expanded is not None, \
                        "expand_multiple_dims() returned `None` on\n" + msg_values

                    r = sess.run(expanded)
                    assert (r == np_tens).all(), "expand_multiple_dims() returned\n{} on\n".format(r) + msg_values

    def test_case_not_tensor(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            for init_num_dims in range(4):
                shape = list(range(1, 1 + init_num_dims))
                size = 1
                if len(shape) > 0:
                    for d in shape:
                        size *= d
                if init_num_dims == 0:
                    values = np.array(1.)
                else:
                    values = np.arange(0, size, 1)
                for _ in range(init_num_dims + 1):
                    shape = shape[-1:] + shape[:-1]
                    np_tens = np.reshape(values, shape)

                    axes = tf.range(init_num_dims)
                    num_dims = tf.constant(init_num_dims)

                    expanded = tensor_ops.expand_multiple_dims(np_tens, num_dims, axes)
                    msg_values = "tensor == {}\n" \
                        "num_dims == {}\n" \
                        "axes == {}".format(np_tens, init_num_dims, list(range(init_num_dims)))
                    assert expanded is not None, \
                        "expand_multiple_dims() returned `None` on\n" + msg_values

                    r = sess.run(expanded)
                    assert (r == np_tens).all(), "expand_multiple_dims() returned\n{} on\n".format(r) + msg_values

    def test_raises_if_num_dims_is_too_small(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            num_dims = tf.placeholder(tf.int32)
            tensor = tf.placeholder(tf.float32)
            axes = tf.placeholder(tf.int32)
            expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
            with pytest.raises(tf.errors.InvalidArgumentError):
                _ = sess.run(expanded, feed_dict={num_dims: 1, tensor: [[2]], axes: [1, 2]})

    def test_raises_on_large_axes_values(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            num_dims = tf.placeholder(tf.int32)
            tensor = tf.placeholder(tf.float32)
            axes = tf.placeholder(tf.int32)
            expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
            with pytest.raises(tf.errors.InvalidArgumentError):
                _ = sess.run(expanded, feed_dict={num_dims: 1, tensor: [1], axes: [1]})

    def test_raises_on_too_small_axes_values(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            num_dims = tf.placeholder(tf.int32)
            tensor = tf.placeholder(tf.float32)
            axes = tf.placeholder(tf.int32)
            expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
            with pytest.raises(tf.errors.InvalidArgumentError):
                _ = sess.run(expanded, feed_dict={tensor: [1], num_dims: 1, axes: [-2]})

    def test_usual_permutation(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tensor_axes_output = [
                (
                    [[1, 2],
                     [3, 4]],

                    [1, 0],

                    [[1, 3],
                     [2, 4]]
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    [1, 0, 2],

                    [[[1, 2],
                      [5, 6]],
                     [[3, 4],
                      [7, 8]]],
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    [0, 2, 1],

                    [[[1, 3],
                      [2, 4]],
                     [[5, 7],
                      [6, 8]]]
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    [2, 1, 0],

                    [[[1, 5],
                      [3, 7]],
                     [[2, 6],
                      [4, 8]]]
                )
            ]
            for tensor, axes, output in tensor_axes_output:
                tensor = np.array(tensor)
                expanded = tensor_ops.expand_multiple_dims(tensor, tensor.ndim, axes)
                r = sess.run(expanded)
                assert (r == np.array(output)).all(), 'failed on tensor={} and axes={}'.format(tensor, axes)

    def test_permutation_with_negative_axes(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tensor_axes_output = [
                (
                    [[1, 2],
                     [3, 4]],

                    [-1, -2],

                    [[1, 3],
                     [2, 4]]
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    [-2, -3, -1],

                    [[[1, 2],
                      [5, 6]],
                     [[3, 4],
                      [7, 8]]],
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    [-3, -1, -2],

                    [[[1, 3],
                      [2, 4]],
                     [[5, 7],
                      [6, 8]]]
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    [-1, -2, -3],

                    [[[1, 5],
                      [3, 7]],
                     [[2, 6],
                      [4, 8]]]
                )
            ]
            for tensor, axes, output in tensor_axes_output:
                tensor = np.array(tensor)
                expanded = tensor_ops.expand_multiple_dims(tensor, tensor.ndim, axes)
                r = sess.run(expanded)
                assert (r == np.array(output)).all(), 'failed on tensor={} and axes={}'.format(tensor, axes)

    def test_insert_dims_before(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tensor_numdims_axes_output = [
                (
                    [1, 2],

                    2,

                    [1],

                    [[1, 2]]
                ),
                (
                    [[1, 2]],

                    3,

                    [1, 2],

                    [[[1, 2]]]
                ),
                (
                    [1, 2],

                    3,

                    [2],

                    [[[1, 2]]]
                ),
            ]
            for tensor, num_dims, axes, output in tensor_numdims_axes_output:
                tensor = np.array(tensor)
                expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
                r = sess.run(expanded)
                assert (r == np.array(output)).all(), \
                    'failed on tensor={}, num_dims={} and axes={}\noutput={}\nexpected={}'.format(
                        tensor, num_dims, axes, r, np.array(output))

    def test_insert_dims_after(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tensor_numdims_axes_output = [
                (
                    [1, 2],

                    2,

                    [0],

                    [[1], [2]]
                ),
                (
                    [[1, 2]],

                    3,

                    [0, 1],

                    [[[1], [2]]]
                ),
                (
                    [1, 2],

                    3,

                    [0],

                    [[[1]], [[2]]]
                ),
            ]
            for tensor, num_dims, axes, output in tensor_numdims_axes_output:
                tensor = np.array(tensor)
                expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
                r = sess.run(expanded)
                assert (r == np.array(output)).all(), \
                    'failed on tensor={}, num_dims={} and axes={}\noutput={}\nexpected={}'.format(
                        tensor, num_dims, axes, r, np.array(output))

    def test_insert_dims_inside(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tensor_numdims_axes_output = [
                (
                    [[1, 2], [3, 4]],

                    3,

                    [0, 2],

                    [[[1, 2]], [[3, 4]]]
                ),
                (
                    [[1, 2], [3, 4]],

                    4,

                    [0, 3],

                    [[[[1, 2]]], [[[3, 4]]]]
                ),
                (
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],

                    5,

                    [0, 2, 4],

                    [[[[[1, 2]], [[3, 4]]]], [[[[5, 6]], [[7, 8]]]]],
                ),
            ]
            for tensor, num_dims, axes, output in tensor_numdims_axes_output:
                tensor = np.array(tensor)
                expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
                r = sess.run(expanded)
                assert (r == np.array(output)).all(), \
                    'failed on tensor={}, num_dims={} and axes={}\noutput={}\nexpected={}'.format(
                        tensor, num_dims, axes, r, np.array(output))

    def test_insert_before_and_permute(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tensor_numdims_axes_output = [
                (
                    [[1, 2],
                     [3, 4]],

                    4,

                    [3, 2],

                    [[[[1, 3],
                       [2, 4]]]]
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    5,

                    [3, 2, 4],

                    [[[[[1, 2],
                        [5, 6]],
                       [[3, 4],
                        [7, 8]]]]],
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    5,

                    [2, 4, 3],

                    [[[[[1, 3],
                        [2, 4]],
                       [[5, 7],
                        [6, 8]]]]]
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    5,

                    [4, 3, 2],

                    [[[[[1, 5],
                        [3, 7]],
                       [[2, 6],
                        [4, 8]]]]]
                )
            ]
            for tensor, num_dims, axes, output in tensor_numdims_axes_output:
                tensor = np.array(tensor)
                expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
                r = sess.run(expanded)
                assert (r == np.array(output)).all(), \
                    'failed on tensor={}, num_dims={} and axes={}\noutput={}\nexpected={}'.format(
                        tensor, num_dims, axes, r, np.array(output))

    def test_insert_after_and_permute(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tensor_numdims_axes_output = [
                (
                    [[1, 2],
                     [3, 4]],

                    4,

                    [1, 0],

                    [[[[1]], [[3]]],
                     [[[2]], [[4]]]]
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    5,

                    [1, 0, 2],

                    [[[[[1]], [[2]]],
                      [[[5]], [[6]]]],
                     [[[[3]], [[4]]],
                      [[[7]], [[8]]]]],
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    5,

                    [0, 2, 1],

                    [[[[[1]], [[3]]],
                      [[[2]], [[4]]]],
                     [[[[5]], [[7]]],
                      [[[6]], [[8]]]]]
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    5,

                    [2, 1, 0],

                    [[[[[1]], [[5]]],
                      [[[3]], [[7]]]],
                     [[[[2]], [[6]]],
                      [[[4]], [[8]]]]]
                )
            ]
            for tensor, num_dims, axes, output in tensor_numdims_axes_output:
                tensor = np.array(tensor)
                expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
                r = sess.run(expanded)
                assert (r == np.array(output)).all(), \
                    'failed on tensor={}, num_dims={} and axes={}\noutput={}\nexpected={}'.format(
                        tensor, num_dims, axes, r, np.array(output))

    def test_insert_inside_and_permute(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tensor_numdims_axes_output = [
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    5,

                    [2, 0, 4],

                    [[[[[1, 2]],
                       [[5, 6]]]],
                     [[[[3, 4]],
                       [[7, 8]]]]],
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    5,

                    [0, 4, 2],

                    [[[[[1, 3]],
                       [[2, 4]]]],
                     [[[[5, 7]],
                       [[6, 8]]]]]
                ),
                (
                    [[[1, 2],
                      [3, 4]],
                     [[5, 6],
                      [7, 8]]],

                    5,

                    [4, 2, 0],

                    [[[[[1, 5]],
                       [[3, 7]]]],
                     [[[[2, 6]],
                       [[4, 8]]]]]
                )
            ]
            for tensor, num_dims, axes, output in tensor_numdims_axes_output:
                tensor = np.array(tensor)
                expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
                r = sess.run(expanded)
                assert (r == np.array(output)).all(), \
                    'failed on tensor={}, num_dims={} and axes={}\noutput={}\nexpected={}'.format(
                        tensor, num_dims, axes, r, np.array(output))

    def test_expansion_of_no_shape(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tensor = 6.
            num_dims = 3
            axes = []
            expected = [[[6.]]]
            expanded = tensor_ops.expand_multiple_dims(tensor, num_dims, axes)
            r = sess.run(expanded)
            assert (r == np.array(expected)).all(), \
                'failed on tensor={}, num_dims={} and axes={}\noutput={}\nexpected={}'.format(
                    tensor, num_dims, axes, r, np.array(expected)
                )
