import numpy as np


def hist_1d_loop_numpy(tensor, num_bins, range_, axis, sample_size):
    ndim = tensor.ndim
    axis %= ndim
    shape = tensor.shape
    idx = 0
    hist = 0
    while idx < shape[axis]:
        sl = [slice(None)]*axis + \
            [slice(idx, idx+sample_size)] + \
            [slice(None)]*(ndim-2-axis)
        sl = tuple(sl)
        t = tensor[sl]
        hist += hist_1d_numpy(t, num_bins, range_, axis)
        idx += sample_size
    return hist


def adjust_histogram_range_for_numpy(range_, num_bins):
    range_ = list(range_)
    h = (range_[1] - range_[0]) / num_bins
    range_[0] += h
    range_[1] -= h
    return range_


def hist_1d_numpy(tensor, num_bins, range_, axis):
    range_ = adjust_histogram_range_for_numpy(range_, num_bins)
    bins = np.histogram_bin_edges(tensor, num_bins-2, range_)
    tensor = np.digitize(tensor, bins)
    tensor = hist_from_nonnegative_ints_numpy(
        tensor,
        axis,
        num_bins
    )
    return tensor


def shift_axis_numpy(
        tensor,
        axis,
        pos,
):
    shape = tensor.shape
    ndim = tensor.ndim
    axis %= ndim
    pos %= ndim
    dims = list(range(ndim))
    if axis == pos:
        return tensor
    if pos > axis:
        perm = dims[:axis] + dims[axis+1:pos+1] + [axis] + dims[pos+1:]
    else:
        perm = dims[:pos] + [axis] + dims[pos:axis] + dims[axis+1:]
    return tensor.transpose(*perm)


def hist_from_nonnegative_ints_numpy(
        tensor,
        axis,
        num_bins
):
    tensor = shift_axis_numpy(tensor, axis, -1)
    shape = tensor.shape
    tensor = tensor.reshape([-1, shape[-1]])
    n = tensor.shape[0]
    shifts = (np.arange(n) * num_bins).reshape([-1, 1])
    tensor += shifts
    tensor = tensor.reshape([-1])
    tensor = np.bincount(tensor, minlength=n*num_bins)
    tensor = tensor.reshape(list(shape[:-1]) + [-1])
    return shift_axis_numpy(tensor, -1, axis)


def self_cross_sum_numpy(
        tensor,
        axis,
):
    axis %= tensor.ndim
    tensor_2 = np.expand_dims(tensor, axis=axis+1)
    tensor = np.expand_dims(tensor, axis=axis)
    return tensor + tensor_2


def self_cross_sum_numpy_with_factors(
        tensor,
        axis,
        f1,
        f2,
):
    axis %= tensor.ndim
    tensor_2 = np.expand_dims(tensor, axis=axis+1)
    tensor = np.expand_dims(tensor, axis=axis)
    mul1 = f1 * tensor
    mul2 = f2 * tensor_2
    s = mul1 + mul2
    return s


def self_cross_hist(
        tensor,
        value_axis,
        cross_axis,
        num_bins,
        range_,
):
    ndim = tensor.ndim
    value_axis %= ndim
    cross_axis %= ndim
    range_ = adjust_histogram_range_for_numpy(range_, num_bins)
    bins = np.histogram_bin_edges(tensor, num_bins-2, range_)
    tensor = np.digitize(tensor, bins)
    tensor = self_cross_sum_numpy_with_factors(tensor, cross_axis, 1, num_bins)
    value_axis += int(value_axis > cross_axis)
    hist = hist_from_nonnegative_ints_numpy(tensor, value_axis, num_bins**2)
    return hist


def get_self_cross_histograms_numpy(
        activations,
        value_axis,
        cross_axis,
        num_bins,
        range_,
        max_sample_size_per_iteration=10**3,
):
    ndim = activations.ndim
    value_axis %= ndim
    cross_axis %= ndim
    shape = activations.shape
    idx = 0
    hist = 0
    while idx < shape[value_axis]:
        sl = [slice(None)]*value_axis + \
            [slice(idx, idx+max_sample_size_per_iteration)] + \
            [slice(None)]*(ndim-2-value_axis)
        sl = tuple(sl)
        tensor = activations[sl]
        hist += self_cross_hist(
            tensor,
            value_axis,
            cross_axis,
            num_bins,
            range_,
        )
        idx += max_sample_size_per_iteration
    return hist


def entropy_MLE_from_hist_numpy(hist, axis, keepdims=False):
    n = np.sum(hist, axis=axis, keepdims=True)
    hist = hist / n
    log_prob = np.log2(hist)
    hist *= log_prob
    hist = np.nan_to_num(hist)
    return -np.sum(hist, axis=axis, keepdims=keepdims)


def entropy_MM_from_hist_numpy(hist, axis, keepdims=False):
    entropy = entropy_MLE_from_hist_numpy(hist, axis, keepdims=True)
    m = np.count_nonzero(hist, axis=axis)
    m = np.expand_dims(m, axis=axis)
    n = np.sum(hist, axis=axis, keepdims=True)
    entropy = entropy + (m - 1) / (2*n)
    if keepdims:
        return entropy
    return np.squeeze(entropy + (m - 1) / (2*n), axis=axis)


def mutual_information_and_min_nonzero_count_numpy(
        activations,
        value_axis,
        cross_axis,
        num_bins,
        range_,
        keepdims=False,
        sample_size_1d=5*10**5,
        sample_size_2d=2*10**5
):
    hist = hist_1d_loop_numpy(activations, num_bins, range_, value_axis, sample_size_1d)
    entropy = entropy_MM_from_hist_numpy(hist, value_axis, keepdims=True)
    entropy_sum = self_cross_sum_numpy(entropy, cross_axis)
    cross_hist = get_self_cross_histograms_numpy(
        activations,
        value_axis,
        cross_axis,
        num_bins,
        range_,
        sample_size_2d
    )
    value_axis += int(value_axis > cross_axis)
    joint_entropy = entropy_MM_from_hist_numpy(cross_hist, value_axis, keepdims=True)
    entropy = entropy_sum - joint_entropy
    m = np.count_nonzero(cross_hist, axis=value_axis)
    min_nonzero = np.min(cross_hist)
    if keepdims:
        return entropy, min_nonzero
    return np.squeeze(entropy, axis=value_axis), min_nonzero
