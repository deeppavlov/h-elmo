from collections.abc import MutableMapping


class NotFoundError(Exception):
    def __init__(self, message, insert_position):
        self.insert_position = insert_position
        super().__init__(message)


def bin_search_and_insert_pos_search(sorted_seq, value, sorting_key=lambda x: x):
    len_ = len(sorted_seq)
    if len_ > 0:
        lo, hi = 0, len_ - 1
        while hi - lo > 0:
            mid = (hi + lo) // 2
            if sorting_key(sorted_seq[mid]) < value:
                lo = mid + 1
            elif sorting_key(sorted_seq[mid]) > value:
                hi = mid - 1
            else:
                return mid
        if sorting_key(sorted_seq[lo]) == value:
            return lo
        if sorting_key(sorted_seq[lo]) > value:
            ins_pos = lo
        else:
            ins_pos = lo + 1
    else:
        ins_pos = 0
    raise NotFoundError(
        "value is not in sequence. "
        "A position in which value has to be inserted is"
        " available in error.insert_position. error.insert_position == {}".format(ins_pos),
        ins_pos
    )


def search_insert_position(sorted_seq, value, sorting_key=lambda x: x):
    try:
        idx = bin_search_and_insert_pos_search(sorted_seq, value, sorting_key)
    except NotFoundError as e:
        ins_pos = e.insert_position
    else:
        ins_pos = idx + 1
    return ins_pos


def sorting_key_zeroth_element(x):
    return x[0]


def sorting_key_float_zeroth_element(x):
    return float(x[0])


class SortedDict(MutableMapping):
    __slots__ = ('_sorting_key', '_elements_sorting_key', '_elements')

    def __init__(self, *args, **kwargs):
        self._sorting_key = str
        self._elements_sorting_key = sorting_key_zeroth_element
        self._elements = []
        if len(args) == 1:
            try:
                iter(args[0])
            except TypeError:
                raise TypeError("args[0] is not iterable")
            try:
                items = args[0].items()
                first_arg = 'dict'
            except AttributeError:
                items = args[0]
                first_arg = 'seq'
            for idx, element in enumerate(items):
                try:
                    iter(element)
                except TypeError:
                    raise TypeError(
                        "elements of args[0] have to be "
                        "iterables of at least 2 elements"
                    )
                counter = 0
                _element = []
                for v in element:
                    _element.append(v)
                    counter += 1
                    if counter >= 2:
                        break
                if counter < 2:
                    if first_arg == 'seq':
                        msg = ("cannot convert dictionary update sequence "
                               "element #{} to a sequence of 2 or "
                               "more elements".format(idx))
                    else:
                        msg = ("broken update dictionary. Dictionary"
                               " element does not have both key and value")
                    raise TypeError(msg)
                self.__setitem__(_element[0], _element[1])
        elif len(args) > 1:
            raise TypeError("SortedDict expected at most 1 positional arguments, got 2")
        for key, value in kwargs.items():
            self.__setitem__(key, value)
        self._sort()

    def _sort(self):
        self._elements.sort(key=self._elements_sorting_key)

    def __len__(self):
        return len(self._elements)

    def __getitem__(self, key):
        try:
            idx = bin_search_and_insert_pos_search(self._elements, self._sorting_key(key), self._elements_sorting_key)
        except NotFoundError:
            raise KeyError("key {} is not in dictionary".format(key))
        return self._elements[idx][1]

    def __setitem__(self, key, value):
        try:
            idx = bin_search_and_insert_pos_search(self._elements, self._sorting_key(key), self._elements_sorting_key)
            self._elements[idx] = [key, value]
        except NotFoundError as e:
            ins_pos = e.insert_position
            self._elements.insert(ins_pos, [key, value])

    def __delitem__(self, key):
        try:
            idx = bin_search_and_insert_pos_search(self._elements, self._sorting_key(key), self._elements_sorting_key)
            del self._elements[idx]
        except NotFoundError:
            raise KeyError("key {} is not in dictionary".format(key))

    def __iter__(self):
        for elem in self._elements:
            yield elem[0]

    def set_sorting_key(self, sorting_key):
        self._sorting_key = sorting_key
        self._sort()

    def get_sorting_key(self):
        return self._sorting_key

    def get_ith_element(self, i):
        return self._elements[i]

    def __repr__(self):
        elements = ', '.join(['({}, {})'.format(repr(k), repr(v)) for k, v in self.items()])
        return '{}([{}])'.format(self.__class__.__name__, elements)

    def __str__(self):
        return repr(self)
