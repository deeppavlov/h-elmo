import numpy as np

import helmo.util.numpy_correlation as correlation


class TestGetCorrelation:
    def test_no_correlation(self):
        tensors = [
            np.random.normal(size=[3, 100000, 5]),
            np.random.normal(size=[100000, 3, 5]),
            np.random.normal(size=[5, 3, 100000]),
        ]
        cor_axes = [-1, -1, 0]
        new_cor_axes = [1, 1, 0]
        reduced_axes_values = [[1], [0], [-1]]
        for tensor, reduced_axes, cor_axis, new_cor_axis in zip(tensors, reduced_axes_values, cor_axes, new_cor_axes):
            corr = correlation.get_correlation(tensor, reduced_axes=reduced_axes, cor_axis=cor_axis)
            abs_not_diag = np.abs(corr[correlation.get_not_diag_mask(corr.shape, new_cor_axis, new_cor_axis+1)])
            epsilon = 0.02
            assert np.all(abs_not_diag < epsilon), \
                "Failed on not correlated tensor of shape {}\nreduced_axes={}\ncor_axis=-1\noutput error={}".format(
                    tensor.shape, reduced_axes, abs_not_diag[abs_not_diag > epsilon]
                )

    def test_full_correlation(self):
        vec = np.random.normal(size=[10**5])
        correlated = np.stack([vec, -vec], axis=1)
        not_correlated = np.random.normal(size=[10**5, 3])
        tensor = np.concatenate([correlated, not_correlated], axis=1)
        tensors = [tensor]
        cor_axes = [-1]
        new_cor_axes = [0]
        reduced_axes_values = [[0]]
        epsilon = 0.02
        for tensor, reduced_axes, cor_axis, new_cor_axis in zip(tensors, reduced_axes_values, cor_axes, new_cor_axes):
            corr = correlation.get_correlation(tensor, reduced_axes=reduced_axes, cor_axis=cor_axis)
            expected = np.array(
                [[1, -1, 0, 0, 0],
                 [-1, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]]
            )
            assert np.all(np.logical_and(expected-epsilon < corr, corr < expected+epsilon)), \
                "Failed on input\ntensor.shape={}\nreduced_axes={}\ncor_axis={}\noutput={}".format(
                    tensor.shape,
                    reduced_axes,
                    cor_axis,
                    corr,
                )


class TestGetMeanCorrelation:
    def test_no_correlation(self):
        tensors = [
            np.random.normal(size=[3, 100000, 5]),
            np.random.normal(size=[100000, 3, 5]),
            np.random.normal(size=[5, 3, 100000]),
        ]
        cor_axes = [-1, -1, 0]
        reduced_axes_values = [[1], [0], [-1]]
        for tensor, reduced_axes, cor_axis in zip(tensors, reduced_axes_values, cor_axes):
            corr = correlation.get_mean_correlation(tensor, reduced_axes=reduced_axes, cor_axis=cor_axis)
            expected = np.array(
                [[1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]]
            )
            epsilon = 0.02
            assert np.all(np.logical_and(expected-epsilon < corr, corr < expected+epsilon)), \
                "Failed on not correlated tensor of shape {}\nreduced_axes={}\ncor_axis=-1\noutput={}".format(
                    tensor.shape, reduced_axes, corr
                )

    def test_full_correlation(self):
        vec = np.random.normal(size=[3, 10**5, 1])
        correlated = np.concatenate([vec, -vec], axis=2)
        not_correlated = np.random.normal(size=[3, 10**5, 3])
        tensor = np.concatenate([correlated, not_correlated], axis=2)
        tensors = [tensor]
        cor_axes = [-1]
        reduced_axes_values = [[1]]
        for tensor, reduced_axes, cor_axis in zip(tensors, reduced_axes_values, cor_axes):
            corr = correlation.get_mean_correlation(tensor, reduced_axes=reduced_axes, cor_axis=cor_axis)
            expected = np.array(
                [[1, -1, 0, 0, 0],
                 [-1, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]]
            )
            epsilon = 0.02
            assert np.all(np.logical_and(expected - epsilon < corr, corr < expected + epsilon)), \
                "Failed on not correlated tensor of shape {}\nreduced_axes={}\ncor_axis=-1\noutput={}".format(
                    tensor.shape, reduced_axes, corr
                )
