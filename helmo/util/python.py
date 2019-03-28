from typing import Dict, List
import copy
import warnings


def is_namedtuple_instance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def filter_dict_by_keys(d: Dict, list_of_keys: List) -> Dict:
    """
    Create a new dictionary from elements of dictionary `d` which
    keys are in list `lk`.

    Args:
        d: a dictionary with elements from which the result is built
        list_of_keys: a list with allowed keys

    Returns:
        A dictionary with keys which are both in `lk` and `d`.
    """
    return d.__class__(filter(lambda x: x[0] in list_of_keys, d.items()))


def decorate_with_post_sort(fn):
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        args[0].sort()
    return wrapper


def get_iterable_length(it):
    return len(list(copy.deepcopy(it)))


def _warn_about_not_equal_iterable_lengths(a, b, msg):
    len_a = get_iterable_length(a)
    len_b = get_iterable_length(b)
    if len_a != len_b:
        warnings.warn(msg + '\nlen(a) == {}\nlen(b) == {}'.format(len_a, len_b))


def subtract_iterable(a, b):
    _warn_about_not_equal_iterable_lengths(
        a, b,
        "iterbale lengths are not equal",
    )
    return [aa - bb for aa, bb in zip(a, b)]


def add_iterable(a, b):
    _warn_about_not_equal_iterable_lengths(
        a, b,
        "iterbale lengths are not equal",
    )
    return [aa + bb for aa, bb in zip(a, b)]


def is_iterable(a):
    try:
        iter(a)
    except TypeError:
        return False
    return True


def add_scalar_to_iterable(it, scalar):
    """
    Create new list in which each element is a sum of element
    of `list_` and `scalar`.
    :param it: iterable of objects supporting __add__ operator
    :param scalar: an object supporting __add__ operator
    :return: new list with summation results
    """
    return [e + scalar for e in it]
