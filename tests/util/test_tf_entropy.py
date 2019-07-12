import random
import sys
import timeit

import numpy as np
import tensorflow as tf

from learning_to_learn import tensors


np.set_printoptions(threshold=sys.maxsize)


def test_PermuteTwoAxes():
    print('\n' + "*" * 20 + '\nPermuteTwoAxes')
    #     a = tf.reshape(tf.range(12), [3, 4])
    a = tf.ones([2, 3, 4, 5])

    with tensors.PermuteTwoAxes(a, -2, axis_2=1) as ctx:
        t = ctx.tensor
        ctx.tensor = tf.reduce_sum(t, axis=-1, keepdims=True)

    t2 = ctx.tensor

    with tf.Session() as sess:
        # print(sess.run(ctx.print))
        print(sess.run([a, t, t2]))


def test_TensorToMatrix():
    print('\n' + "*" * 20 + '\nTensorToMatrix')
    a = tf.reshape(tf.range(12), [3, 4])
    with tensors.TensorToMatrix(a) as ctx:
        t = ctx.tensor
        ctx.tensor = tf.reduce_sum(t, axis=-1, keepdims=True)

    t2 = ctx.tensor
    with tf.Session() as sess:
        # print(sess.run(ctx.print))
        print(sess.run([a, t, t2]))


def test_hist_1d():
    print('\n' + "*" * 20 + '\nhist_1d')
    a = tf.truncated_normal([5, 1000, 2], mean=1, stddev=1)
    num_bins = 10
    range_ = [-4., 4.]
    axis = -2
    histograms = tensors.hist_1d(a, num_bins, range_, axis)
    with tf.Session() as sess:
        print(sess.run([histograms]))


def test_compute_probabilities():
    print('\n' + "*" * 20 + '\ncompute_probabilities')
    a = tf.truncated_normal([5, 1000, 2], mean=1, stddev=1)
    num_bins = 10
    range_ = [-4., 4.]
    axis = -2
    probabilities = tensors.compute_probabilities(a, num_bins, range_, axis)
    with tf.Session() as sess:
        print(sess.run([probabilities]))


def test_entropy_MM_from_prob():
    print('\n' + "*" * 20 + '\nentropy_MM_from_prob')
    axis = -2
    probabilities = tf.constant(
        [[[0.3], [0.3], [0.3]],
         [[.5], [.5], [0.]],
         [[.1], [.1], [.8]]]
    )
    n = 100
    shape = tf.shape(probabilities)
    m = [[[3]], [[2]], [[3]]]
    entropy = tensors.entropy_MM_from_prob(probabilities, n, m, axis)
    with tf.Session() as sess:
        print(sess.run([entropy]))


def test_entropy_MLE_from_prob():
    print('\n' + "*" * 20 + '\nentropy_MLE_from_prob')
    axis = -2
    probabilities = tf.constant(
        [[[0.3], [0.3], [0.3]],
         [[1.], [0.], [0.]],
         [[.1], [.1], [.8]]]
    )
    entropy = tensors.entropy_MLE_from_prob(probabilities, axis)
    with tf.Session() as sess:
        res = sess.run(entropy)
        print(res)


def test_entropy_MLE_from_hist():
    print('\n' + "*" * 20 + '\nentropy_MLE_from_hist')
    axis = 1
    hist = tf.constant(
        [[[1], [1], [1]],
         [[2], [2], [2]],
         [[1], [2], [3]],
         [[10], [10000], [10]],
         [[0], [0], [1]]]
    )
    entropy = tensors.entropy_MLE_from_hist(hist, axis)
    with tf.Session() as sess:
        res = sess.run(entropy)
        print(res)


def test_entropy_MM_from_hist():
    print('\n' + "*" * 20 + '\nentropy_MM_from_hist')
    axis = -2
    hist = tf.constant(
        [[[1], [1], [1]],
         [[2], [2], [2]],
         [[1], [2], [3]],
         [[10], [10000], [10]],
         [[0], [0], [1]]]
    )
    entropy = tensors.entropy_MM_from_hist(hist, axis)
    with tf.Session() as sess:
        res = sess.run(entropy)
        print(res)


def test_mean_neuron_entropy_with_digitalization():
    print('\n' + "*" * 20 + '\nmean_neuron_entropy_with_digitalization')
    a = tf.truncated_normal([10, 1000])
    axis = 1
    mean_entropy = tensors.mean_neuron_entropy_with_digitalization(a, axis, 100, [-3., 3.])
    with tf.Session() as sess:
        res = sess.run(mean_entropy)
        print(res)


def test2_entropy_MM_from_hist():
    print('\n' + "*" * 20 + '\nentropy_MM_from_hist')
    axis = -1
    a = tf.truncated_normal([10, 1000])
    hist = tensors.hist_1d(a, 100, [-1., 1.], axis)
    entropy = tensors.entropy_MM_from_hist(hist, axis)
    with tf.Session() as sess:
        res = sess.run(entropy)
        print(res)


def test_shift_axis():
    tensor = tf.zeros([2, 3, 4])
    t = tensors.shift_axis(tensor, 0, 2)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(sess.run(t))


def test_get_output_shape_for_hist_1d():
    print('\n' + "*" * 20 + '\nget_output_shape_for_hist_1d')
    tensor = tf.zeros([2, 3, 4, 5, 6, 7])
    axis = -1
    num_bins = 10
    shape = tensors.get_output_shape_for_hist_1d(tensor, axis, num_bins)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(sess.run(shape))


def test_self_cross_sum_with_factors():
    print('\n' + "*" * 20 + '\nself_cross_sum_with_factors')
    tensor = [[1, 2, 3], [4, 5, 6]]
    cs = tensors.self_cross_sum_with_factors(tensor, 0, 10, 1)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(sess.run(cs))


def test_hist_from_nonnegative_ints():
    print('\n' + "*" * 20 + '\nhist_from_nonnegative_ints')
    tensor = tf.histogram_fixed_width_bins(
        tf.random.normal([100000], dtype=tf.float32, mean=0, stddev=1),
        [-15., 15.],
        nbins=30,
    )
    tensor = tf.reshape(tensor, [10000, 10])
    hist = tensors.hist_from_nonnegative_ints(tensor, -2, 30)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(np.max(sess.run(tensor)))
        print(sess.run(hist))


def test_cross_hist_from_tensor():
    print('\n' + "*" * 20 + '\ncross_hist_from_tensor')
    tensor = [[1. for _ in range(20)] + [2. for _ in range(20)], [3. for _ in range(20)] + [4. for _ in range(20)]]
    hist = tensors.cross_hist_from_tensor(tensor, 10, [0., 10.])
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(sess.run(hist))


def test_add_cross_hist_1_slice_independent():
    print('\n' + "*" * 20 + '\nadd_cross_hist_1_slice_independent')
    idx = tf.constant(3)
    histograms = tf.zeros([2, 2, 100], dtype=tf.int32)
    activations = tf.constant(
        [
            [
                [1. for _ in range(20)] + [2. for _ in range(20)] + \
                [1. for _ in range(20)],
                [3. for _ in range(20)] + [4. for _ in range(20)] + \
                [3. for _ in range(10)] + [4. for _ in range(10)]
            ]
        ]
    )
    num_bins = 10
    range_ = [0., 10.]
    max_sample_size_per_iteration = 10
    idx1, histograms1 = tensors.add_cross_hist_1_slice_independent(
        idx,
        histograms,
        activations,
        num_bins,
        range_,
        max_sample_size_per_iteration
    )
    idx2, histograms2 = tensors.add_cross_hist_1_slice_independent(
        idx1,
        histograms1,
        activations,
        num_bins,
        range_,
        max_sample_size_per_iteration
    )
    idx3, histograms3 = tensors.add_cross_hist_1_slice_independent(
        idx2,
        histograms2,
        activations,
        num_bins,
        range_,
        max_sample_size_per_iteration
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(sess.run([idx1, histograms1]))
        print(sess.run([idx2, histograms2]))
        print(sess.run([idx3, histograms3]))


def test_get_init_shape_of_cross_histograms():
    print('\n' + "*" * 20 + '\nget_init_shape_of_cross_histograms')
    activations = tf.zeros([5, 6, 7, 8])
    num_bins = 10
    shape = tensors.get_init_shape_of_cross_histograms(activations, num_bins)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(sess.run(shape))


def test_sum_self_cross_histograms():
    print('\n' + "*" * 20 + '\nsum_self_cross_histograms')
    activations = tf.constant(
        [
            [
                [1. for _ in range(20)] + [2. for _ in range(20)] + \
                [1. for _ in range(20)],
                [3. for _ in range(20)] + [4. for _ in range(20)] + \
                [3. for _ in range(10)] + [4. for _ in range(10)]
            ]
        ]
    )
    num_bins = 10
    range_ = [0., 10.]
    max_sample_size_per_iteration = 10
    histograms = tensors.sum_self_cross_histograms(
        activations,
        num_bins,
        range_,
        max_sample_size_per_iteration,
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(sess.run(histograms))


def test_get_cross_histograms_permutation():
    print('\n' + "*" * 20 + '\nget_cross_histograms_permutation')
    perm = tensors.get_cross_histograms_permutation(10, 9, 8)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(sess.run(perm))


def test_get_self_cross_histograms():
    print('\n' + "*" * 20 + '\nget_self_cross_histograms')
    activations = tf.constant(
        [
            [
                [1. for _ in range(20)] + [2. for _ in range(20)] + \
                [1. for _ in range(20)],
                [3. for _ in range(20)] + [4. for _ in range(20)] + \
                [3. for _ in range(10)] + [4. for _ in range(10)]
            ]
        ]
    )
    #     activations = tf.transpose(
    #         activations,
    #         perm=[2, 1, 0]
    #     )
    #     value_axis = 0
    #     cross_axis = 1
    value_axis = -1
    cross_axis = -2
    num_bins = 5
    range_ = [0., 5.]
    max_sample_size_per_iteration = 10
    print(activations.get_shape().as_list())
    histograms = tensors.get_self_cross_histograms(
        activations,
        value_axis,
        cross_axis,
        num_bins,
        range_,
        max_sample_size_per_iteration,
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        h = sess.run(histograms)[0]
        print(h.shape)
        print(h)


def test_get_min_nonzero():
    print('\n' + "*" * 20 + '\ntest_get_min_nonzero')
    tensor = tf.concat([tf.zeros([2, 3, 4]), tf.ones([2, 3, 4]), 2 * tf.ones([2, 3, 4]), 3 * tf.ones([2, 3, 4])], 0)
    nz = tensors.get_min_nonzero(tensor)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print(sess.run([nz]))


def test_mutual_information_and_min_nonzero_count():
    print('\n' + "*" * 20 + '\ntest_mutual_information_and_min_nonzero_count')
    with tf.device('/gpu:0'):
        #         rand_vec = tf.random_normal([1, 100000000])
        #         rand_vec_2 = tf.random_normal([1, 100000000])
        #         max_mutual_info_activations = tf.concat([rand_vec, -rand_vec], 0)
        #         min_mutual_info_activations = tf.concat([rand_vec, rand_vec_2], 0)
        real_activatons = tf.random_normal([10, 16 * 10 ** 6])
        activations = real_activatons
        value_axis = -1
        cross_axis = -2
        num_bins = 100
        range_ = [-2., 2.]
        mutual_info, min_num_events = tensors.mutual_information_and_min_nonzero_count(
            activations,
            value_axis,
            cross_axis,
            num_bins,
            range_,
            max_sample_size_per_iteration=2 * 10 ** 4,
        )
        hists = tensors.hist_1d_loop(activations, num_bins, [-2., 2.], -1, 10 ** 6)
        entropy = tensors.entropy_MM_from_hist(hists, value_axis)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    with tf.Session(config=config) as sess:
        entropy, mutual_info, min_num_events = sess.run(
            [entropy, mutual_info, min_num_events],
            options=run_options,
        )
        print('entropy:\n', entropy)
        print('\nmutual_info:\n', mutual_info)
        print('\nmin_num_events:\n', min_num_events)


def test_get_slice_specs():
    print('\n' + "*" * 20 + '\ntest_get_slice_specs')
    shape = [3, 4, 5, 6]
    idx = 2
    axis = -1
    sample_size = 10
    start, size = tensors.get_slice_specs(shape, axis, idx, sample_size)
    with tf.Session() as sess:
        print(sess.run([start, size]))


def test_hist_1d_loop():
    print('\n' + "*" * 20 + '\ntest_hist_1d_loop')
    values = tf.random_normal([10, 1000, 3])
    num_bins = 20
    range_ = [-3., 3.]
    axis = 1
    max_sample_size_per_iteration = 10
    hist = tensors.hist_1d_loop(values, num_bins, range_, axis, max_sample_size_per_iteration)
    with tf.Session() as sess:
        print(sess.run(hist))


def test_sample_from_distribution():
    distr = [.25, .25, .25, .25, .0]
    l = [tensors.sample_from_distribution(distr) for _ in range(100)]
    print(l)


def test_sample_from_distribution_continuous():
    distr = [.0, .25, .25, .25, .25]
    borders = [[float(i), float(i + 1)] for i in range(5)]
    l = [tensors.sample_from_distribution_continuous(distr, borders) for _ in range(100)]
    print(l)


def test_mean_mutual_information_and_min_nonzero_count():
    print('\n' + "*" * 20 + '\ntest_mean_mutual_information_and_min_nonzero_count')
    with tf.device('/gpu:0'):
        #         rand_vec = tf.random_normal([1, 100000000])
        #         rand_vec_2 = tf.random_normal([1, 100000000])
        #         max_mutual_info_activations = tf.concat([rand_vec, -rand_vec], 0)
        #         min_mutual_info_activations = tf.concat([rand_vec, rand_vec_2], 0)
        real_activatons = tf.random_normal([10, 16 * 10 ** 5])
        activations = real_activatons
        value_axis = -1
        cross_axis = -2
        num_bins = 100
        range_ = [-2., 2.]
        mutual_info, mean_mutual_info, min_num_events = tensors.mean_mutual_information_and_min_nonzero_count(
            activations,
            value_axis,
            cross_axis,
            num_bins,
            range_,
            max_sample_size_per_iteration=2 * 10 ** 4,
        )
        hists = tensors.hist_1d_loop(activations, num_bins, range_, value_axis, 10 ** 6)
        entropy = tensors.entropy_MM_from_hist(hists, value_axis)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    with tf.Session(config=config) as sess:
        entropy, mutual_info, mean_mutual_info, min_num_events = sess.run(
            [entropy, mutual_info, mean_mutual_info, min_num_events],
            options=run_options,
        )
        print('entropy:\n', entropy)
        print('\nmutual_info:\n', mutual_info)
        print('\nmean_mutual_info:\n', mean_mutual_info)
        print('\nmin_num_events:\n', min_num_events)


def test_mean_mutual_information_and_min_nonzero_count_2():
    print('\n' + "*" * 20 + '\ntest_mean_mutual_information_and_min_nonzero_count')
    distr = [.0, .25, .25, .25, .25]
    borders = [[float(i), float(i + 1)] for i in range(5)]
    with tf.device('/gpu:0'):
        #         rand_vec = tf.random_normal([1, 100000000])
        #         rand_vec_2 = tf.random_normal([1, 100000000])
        #         max_mutual_info_activations = tf.concat([rand_vec, -rand_vec], 0)
        #         min_mutual_info_activations = tf.concat([rand_vec, rand_vec_2], 0)
        activations = tf.placeholder(tf.float32)
        value_axis = -2
        cross_axis = -1
        num_bins = 5
        range_ = [-0., 5.]
        mutual_info, mean_mutual_info, min_num_events = tensors.mean_mutual_information_and_min_nonzero_count(
            activations,
            value_axis,
            cross_axis,
            num_bins,
            range_,
            max_sample_size_per_iteration=2 * 10 ** 4,
        )
        hists = tensors.hist_1d_loop(activations, num_bins, range_, value_axis, 10 ** 6)
        entropy = tensors.entropy_MM_from_hist(hists, value_axis)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    with tf.Session(config=config) as sess:
        for i in [50, 100, 200, 400, 1000, 2000, 5000, 10000]:
            real_activations = np.array(
                [
                    [tensors.sample_from_distribution_continuous(distr, borders) for _ in range(i)]
                    for _ in range(4)
                ]
            ).transpose()
            entropy_r, mutual_info_r, mean_mutual_info_r, min_num_events_r = sess.run(
                [entropy, mutual_info, mean_mutual_info, min_num_events],
                options=run_options,
                feed_dict={activations: real_activations}
            )
            print('\n' + '#' * 20)
            print("number of points:", i)
            print('entropy:\n', entropy_r)
            print('\nmutual_info:\n', mutual_info_r)
            print('\nmean_mutual_info:\n', mean_mutual_info_r)
            print('\nmin_num_events:\n', min_num_events_r)


def test_mean_mutual_information_and_min_nonzero_count_3():
    print('\n' + "*" * 20 + '\ntest_mean_mutual_information_and_min_nonzero_count')
    distr = [.0, .25, .25, .25, .25]
    borders = [[float(i), float(i + 1)] for i in range(5)]
    with tf.device('/gpu:0'):
        #         rand_vec = tf.random_normal([1, 100000000])
        #         rand_vec_2 = tf.random_normal([1, 100000000])
        #         max_mutual_info_activations = tf.concat([rand_vec, -rand_vec], 0)
        #         min_mutual_info_activations = tf.concat([rand_vec, rand_vec_2], 0)
        activations = tf.placeholder(tf.float32)
        value_axis = -2
        cross_axis = -1
        num_bins = 5
        range_ = [-0., 5.]
        mutual_info, mean_mutual_info, min_num_events = tensors.mean_mutual_information_and_min_nonzero_count(
            activations,
            value_axis,
            cross_axis,
            num_bins,
            range_,
            max_sample_size_per_iteration=2 * 10 ** 4,
        )
        hists = tensors.hist_1d_loop(activations, num_bins, range_, value_axis, 10 ** 6)
        entropy = tensors.entropy_MM_from_hist(hists, value_axis)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    with tf.Session(config=config) as sess:
        for i in [50, 100, 200, 400, 1000, 2000, 5000, 10000]:
            real_activations = np.array(
                [
                    [tensors.sample_from_distribution_continuous(distr, borders) for _ in range(i)]
                    for _ in range(2)
                ] + [[tensors.sample_from_distribution_continuous(distr, borders) for _ in range(i)]] * 2
            ).transpose()
            entropy_r, mutual_info_r, mean_mutual_info_r, min_num_events_r = sess.run(
                [entropy, mutual_info, mean_mutual_info, min_num_events],
                options=run_options,
                feed_dict={activations: real_activations}
            )
            print('\n' + '#' * 20)
            print("number of points:", i)
            print('entropy:\n', entropy_r)
            print('\nmutual_info:\n', mutual_info_r)
            print('\nmean_mutual_info:\n', mean_mutual_info_r)
            print('\nmin_num_events:\n', min_num_events_r)


def test_mean_mutual_information_and_min_nonzero_count_4():
    print('\n' + "*" * 20 + '\ntest_mean_mutual_information_and_min_nonzero_count')
    distr = [.0, .25, .25, .25, .25]
    borders = [[float(i), float(i + 1)] for i in range(5)]
    with tf.device('/gpu:0'):
        #         rand_vec = tf.random_normal([1, 100000000])
        #         rand_vec_2 = tf.random_normal([1, 100000000])
        #         max_mutual_info_activations = tf.concat([rand_vec, -rand_vec], 0)
        #         min_mutual_info_activations = tf.concat([rand_vec, rand_vec_2], 0)
        activations = tf.placeholder(tf.float32)
        value_axis = -2
        cross_axis = -1
        num_bins = 5
        range_ = [-0., 5.]
        mutual_info, mean_mutual_info, min_num_events = tensors.mean_mutual_information_and_min_nonzero_count(
            activations,
            value_axis,
            cross_axis,
            num_bins,
            range_,
            max_sample_size_per_iteration=2 * 10 ** 4,
        )
        hists = tensors.hist_1d_loop(activations, num_bins, range_, value_axis, 10 ** 6)
        entropy = tensors.entropy_MM_from_hist(hists, value_axis)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    with tf.Session(config=config) as sess:
        for i in [50, 100, 200, 400, 1000, 2000, 5000, 10000]:
            real_activations = 0.5 * np.array(
                [
                    [tensors.sample_from_distribution_continuous(distr, borders) for _ in range(i)]
                    for _ in range(4)
                ] + np.array([[tensors.sample_from_distribution_continuous(distr, borders) for _ in range(i)]] * 4)
            ).transpose()
            entropy_r, mutual_info_r, mean_mutual_info_r, min_num_events_r = sess.run(
                [entropy, mutual_info, mean_mutual_info, min_num_events],
                options=run_options,
                feed_dict={activations: real_activations}
            )
            print('\n' + '#' * 20)
            print("number of points:", i)
            print('entropy:\n', entropy_r)
            print('\nmutual_info:\n', mutual_info_r)
            print('\nmean_mutual_info:\n', mean_mutual_info_r)
            print('\nmin_num_events:\n', min_num_events_r)


if __name__ == '__main__':
    test_PermuteTwoAxes()
    test_TensorToMatrix()
    test_hist_1d()
    test_compute_probabilities()
    test_entropy_MLE_from_prob()
    test_entropy_MM_from_prob()
    test_entropy_MLE_from_hist()
    test_entropy_MM_from_hist()
    test_mean_neuron_entropy_with_digitalization()
    test2_entropy_MM_from_hist()
    test_shift_axis()
    test_get_output_shape_for_hist_1d()
    test_self_cross_sum_with_factors()
    test_hist_from_nonnegative_ints()
    test_cross_hist_from_tensor()
    test_add_cross_hist_1_slice_independent()
    test_get_init_shape_of_cross_histograms()
    test_sum_self_cross_histograms()
    test_get_cross_histograms_permutation()
    test_get_self_cross_histograms()
    test_get_min_nonzero()
    test_sample_from_distribution()
    test_sample_from_distribution_continuous()
    test_mean_mutual_information_and_min_nonzero_count()
    test_mean_mutual_information_and_min_nonzero_count_4()
    test_mean_mutual_information_and_min_nonzero_count_3()
    test_mean_mutual_information_and_min_nonzero_count_2()

    t = timeit.timeit(
        stmt="test_mutual_information_and_min_nonzero_count()",
        globals=dict(
            test_mutual_information_and_min_nonzero_count= \
                test_mutual_information_and_min_nonzero_count
        ),
        number=1,
    )
    print(t)

    test_get_slice_specs()
    test_hist_1d_loop()
