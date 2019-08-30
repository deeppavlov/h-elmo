import numpy as np


def get_self_outer_product(tensor, axis):
    t1 = np.expand_dims(tensor, axis)
    t2 = np.expand_dims(tensor, axis+1)
    return t1 * t2


def shift_reduced_axes_if_required(reduced_axes, covcor_axis):
    return tuple(a+1 if a > covcor_axis else a for a in reduced_axes)


def get_covariance(tensor, reduced_axes, cov_axis, keepdims=False):
    cov_axis %= tensor.ndim
    reduced_axes = [a % tensor.ndim for a in reduced_axes]
    mean = np.mean(tensor, axis=tuple(reduced_axes), keepdims=True)
    devs = tensor - mean
    dev_products = get_self_outer_product(devs, cov_axis)
    reduced_axes = shift_reduced_axes_if_required(reduced_axes, cov_axis)
    return np.mean(dev_products, axis=reduced_axes, keepdims=keepdims)


def get_correlation(tensor, reduced_axes, cor_axis, keepdims=False):
    cor_axis %= tensor.ndim
    reduced_axes = tuple(a % tensor.ndim for a in reduced_axes)
    covariance = get_covariance(tensor, reduced_axes, cor_axis, keepdims=True)
    variance = np.std(tensor, axis=reduced_axes, keepdims=True, ddof=0)
    variance_product = get_self_outer_product(variance, cor_axis)
    correlation = covariance / np.sqrt(variance_product)
    if keepdims:
        return correlation
    reduced_axes = shift_reduced_axes_if_required(reduced_axes, cor_axis)
    return np.squeeze(correlation, axis=reduced_axes)


def get_new_cor_axis(cor_axis, reduced_axes, nd):
    cor_axis %= nd
    reduced_axes = [a % nd for a in reduced_axes]
    new_cor_axis = cor_axis
    for a in reduced_axes:
        if a < cor_axis:
            new_cor_axis -= 1
    return new_cor_axis


def get_axes_for_reduction(tensor, kept_axes):
    kept_axes = [a % tensor.ndim for a in kept_axes]
    reduced_axes = []
    for i in range(tensor.ndim):
        if i not in kept_axes:
            reduced_axes.append(i)
    return reduced_axes


def get_mean_correlation(tensor, reduced_axes, cor_axis):
    cor_axis %= tensor.ndim
    correlation = get_correlation(tensor, reduced_axes, cor_axis)
    new_cor_axis = get_new_cor_axis(cor_axis, reduced_axes, tensor.ndim)
    axes_for_reduction = get_axes_for_reduction(tensor, [new_cor_axis, new_cor_axis+1])
    return np.mean(correlation, axis=tuple(axes_for_reduction))


def range_along_axis(shape, axis):
    shape = tuple(shape)
    nd = len(shape)
    axis %= nd
    range_dim = shape[axis]
    broadcast_shape = shape[:axis] + shape[axis+1:] + (range_dim,)
    r = np.arange(range_dim)
    tensor = np.broadcast_to(r, broadcast_shape)
    perm = list(range(axis)) + [nd-1] + list(range(axis, nd-1))
    return np.transpose(tensor, axes=tuple(perm))


def get_not_diag_mask(shape, axis1, axis2):
    t1 = range_along_axis(shape, axis1)
    t2 = range_along_axis(shape, axis2)
    return np.not_equal(t1, t2)
