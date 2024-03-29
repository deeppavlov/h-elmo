import timeit

from helmo.util.numpy_entropy import *


def test_shift_axis_numpy():
    a = np.zeros((3, 10, 2))
    a = shift_axis_numpy(a, 1, 2)
    print(a)


def test_hist_from_nonnegative_ints_numpy():
    a = np.random.randint(0, 10, size=(3, 10, 2))
    hist = hist_from_nonnegative_ints_numpy(a, 1, 10)
    print(hist)


def test_hist_1d_numpy():
    a = np.random.normal(size=[3, 10000, 2])
    hist = hist_1d_numpy(a, 30, [-3., 3.], 1)
    print(hist)


def test_self_cross_sum_numpy():
    a = np.tile(np.arange(4).reshape([1, -1]), (3, 1))
    a = self_cross_sum_numpy(a, 0)
    print(a)


def test_self_cross_sum_numpy_with_factors():
    a = np.arange(4)
    a = self_cross_sum_numpy_with_factors(a, 0, 10, 1)
    print(a)


def test_self_cross_hist():
    a = np.random.normal(size=[10 ** 4, 4])
    hist = self_cross_hist(a, 0, 1, 10, [-3., 3.])
    print(hist)


def test_get_self_cross_histograms_numpy():
    a = np.random.normal(size=[4, 10 ** 4])
    params = (a, 1, 0, 10, [-3., 3.])
    hist = get_self_cross_histograms_numpy(*params, 100)
    hist_simple = self_cross_hist(*params)
    print(np.all(hist == hist_simple))
    print(hist)


def test_entropy_MLE_from_hist_numpy():
    hist = np.random.randint(0, 100, size=[10000, 4])
    entropy = entropy_MLE_from_hist_numpy(hist, 0, keepdims=True)
    print(entropy)


def test_entropy_MM_from_hist_numpy():
    hist = np.random.randint(0, 100, size=[10000, 4])
    entropy = entropy_MM_from_hist_numpy(hist, 0, keepdims=True)
    print(entropy)


def test_mutual_information_and_min_nonzero_count_numpy():
    activations = np.random.normal(size=[10, 16 * 10 ** 5])
    mi, mnz = mutual_information_and_min_nonzero_count_numpy(
        activations,
        -1, 0,
        100,
        [-2., 2.],
    )
    print(mnz)
    print(mi)


def test_hist_1d_loop_numpy():
    a = np.random.normal(size=[3, 10000, 2])
    hist = hist_1d_loop_numpy(a, 100, [-2., 2.], 1, 100)
    print(hist)


def test_mutual_information_from_hist():
    sets_of_args = [
        (
            np.full([100, 10], 100),
            np.stack(
                [np.diag([100 for _ in range(10)]).reshape([-1]) for _ in range(10**4)]).reshape([100, 100, -1]),
            0,
            -1,
            -1,
            'MLE'
        ),
        (
            np.full([100, 10], 100),
            np.stack(
                [np.diag([100 for _ in range(10)]).reshape([-1]) for _ in range(10 ** 4)]).reshape([100, 100, -1]),
            0,
            -1,
            -1,
            'MM'
        ),
        (
            np.full([100, 10], 100),
            np.full([100, 100, 100], 10),
            0,
            -1,
            -1,
            'MLE'
        ),
        (
            np.full([100, 10], 100),
            np.full([100, 100, 100], 10),
            0,
            -1,
            -1,
            'MM'
        )
    ]
    answers = [
        np.full([100, 100], np.log2(10)),
        # Because empty bins are excluded from support size
        np.full([100, 100], np.log2(10) + 0.0045),
        np.zeros([100, 100]),
        np.full([100, 100], -0.04049999999999976)
    ]
    for i, (args, ans) in enumerate(zip(sets_of_args, answers)):
        mi = mutual_information_from_hist(*args)
        assert np.all(mi == ans), 'test #{}'.format(i)


def test_mean_without_diag():
    inputs = [
        ([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], [-2, -1]),
        ([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [1, 2]),
        ([[[1, 2], [2, 1]], [[1, 4], [4, 1]]], [1, 2]),
    ]
    answers = [0, 1, 3]
    for i, (inp, ans) in enumerate(zip(inputs, answers)):
        assert ans == mean_without_diag(*inp), "failed test #{}".format(i)


if __name__ == '__main__':
    test_shift_axis_numpy()
    test_hist_from_nonnegative_ints_numpy()
    test_hist_1d_numpy()
    test_self_cross_sum_numpy()
    test_self_cross_sum_numpy_with_factors()
    test_get_self_cross_histograms_numpy()
    test_entropy_MLE_from_hist_numpy()
    test_entropy_MM_from_hist_numpy()

    t = timeit.timeit(
        stmt="test_mutual_information_and_min_nonzero_count_numpy()",
        globals=dict(
            test_mutual_information_and_min_nonzero_count_numpy=test_mutual_information_and_min_nonzero_count_numpy
        ),
        number=1
    )
    print(t)

    test_hist_1d_loop_numpy()
    test_mutual_information_from_hist()
    test_mean_without_diag()
