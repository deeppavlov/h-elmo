from typing import Dict, List
from collections import OrderedDict


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
    return {k: v for k, v in d.items() if k in list_of_keys}


def decorate_with_post_sort(fn):
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        args[0].sort()
    return wrapper
