def get_int_part(n):
    return str(int(n // 1))


def get_frac_part(n):
    removed = int(get_int_part(n))
    frac_part = ''
    while n % 1:
        n *= 10
        removed *= 10
        frac_part += str(int(n // 1) - removed)
        removed = int(n // 1)
    return frac_part


def get_kth_digit(number, k, default='0'):
    """Returns k-th digit. For example, in number 123.45 1
    is 2nd digit, 3 is zeroth and 5 is -2nd.
    If the number does not have such a digit default is returned.
    Args:
        number: float or str convertable to float
        k: integer
        default: str
    Returns:
        str"""
    if isinstance(number, str):
        number = float(number)
    int_part = get_int_part(number)
    frac_part = get_frac_part(number)
    number = int_part + frac_part
    k = len(int_part) - k - 1
    if 0 <= k < len(number):
        return number[k]
    else:
        return default


def get_first_nonzero_digit_pos(n):
    if n == 0:
        return None
    int_part = get_int_part(n)
    frac_part = get_frac_part(n)
    if int(int_part):
        return len(int_part) - 1
    i = 0
    while i < len(frac_part) and not int(frac_part[i]):
        i += 1
    assert frac_part[i] != '0'
    return -i - 1


def get_stddev_rounding_num_digits(std, std_acc):
    if std == 0:
        return None

    std_err = std * std_acc

    nz_err = get_first_nonzero_digit_pos(std_err)

    digit_1_pos_higher = get_kth_digit(std, nz_err + 1)

    higher_digit_change = get_kth_digit(std + std_err, nz_err + 1) != digit_1_pos_higher or \
                          get_kth_digit(std - std_err, nz_err + 1) != digit_1_pos_higher

    if higher_digit_change:
        nz_err += 1
    return nz_err


def custom_round(number, digit_num):
    if digit_num is None:
        return number
    if digit_num >= 0:
        exp = 10**digit_num
        remainder = number % exp
        n = number - remainder
        if remainder >= exp / 2:
            return n + exp
        return n
    else:
        return round(number, -digit_num)


def round_mean_and_std(mean, std, std_acc):
    nd = get_stddev_rounding_num_digits(std, std_acc)
    return custom_round(mean, nd), custom_round(std, nd), nd


def create_plus_minus_str(mean, error, nd):
    if nd < 0:
        nd = -nd
    return ("{:.%sf} ± {:.%sf}" % (nd, nd)).format(mean, error)


def create_plus_minus_str_full(mean, std, std_acc):
    mean, std, nd = round_mean_and_std(mean, std, std_acc)
    return create_plus_minus_str(mean, std, nd)
