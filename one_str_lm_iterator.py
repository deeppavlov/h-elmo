from random import Random
from typing import Dict, Tuple, List, Generator, Any, Union
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.common.registry import register


class CursorOutOfBoundsError(Exception):
    def __init__(self, data_len, cursor, msg):
        self.data_len = data_len
        self.cursor = cursor
        self.message = msg


class WrongCursorOrder(Exception):
    def __init__(self, c1, c2, loop, msg):
        self.c1 = c1
        self.c2 = c2
        self.loop = loop
        self.message = msg


@register('one_str_lm_iterator')
class OneStrLmIterator(DataLearningIterator):

    def __init__(self, data: Dict[str, str],
                 seed: int = None, shuffle: bool = False, start_char: str = '\n', no_intersections_in_epoch=False,
                 verbose: bool = False,
                 *args, **kwargs) -> None:
        """ Dataiterator takes a dict with fields 'train', 'test', 'valid'. A list of samples
         (pairs x, y) is stored in each field.
        Args:
            data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y
            can be a tuple of different input features.
            seed (int): random seed for data shuffling. Defaults to None
            shuffle: whether to shuffle data when batching (from config)
        """
        self.shuffle = shuffle

        self.random = Random(seed)

        self._start_char = start_char

        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }
        self._no_intersections_in_epoch = no_intersections_in_epoch
        self._verbose = verbose

    @staticmethod
    def _compute_interval(c1, c2, data_len, loop=False):
        if c1 < 0 or c1 >= data_len:
            raise CursorOutOfBoundsError(
                data_len,
                c1,
                "Cursor c1 = %s has to belong to [0, %s]" % (c1, data_len)
            )
        if c2 < 0 or c2 >= data_len:
            raise CursorOutOfBoundsError(
                data_len,
                c2,
                "Cursor c2 = %s has to belong to [0, %s]" % (c2, data_len)
            )
        if loop:
            length = data_len - c1 + c2
        else:
            length = c2 - c1
        if length < 0:
            raise WrongCursorOrder(
                c1,
                c2,
                loop,
                "Cursors have satisfy condition c1 <= c2 or loop option has to be set"
                " whereas c1 = %s, c2 = %s, loop = %s" % (c1, c2, loop),
            )
        return (c1, c2), length

    def _get_intervals_between_cursors(self, data_len, cursors):
        intervals = list()
        for idx in range(len(cursors)-1):
            intervals.append(
                self._compute_interval(cursors[idx], cursors[idx+1], data_len)
            )
        intervals.append(
            (
                self._compute_interval(cursors[-1], cursors[0], data_len, loop=True)
            )
        )
        intervals = sorted(intervals, key=lambda x: x[1]*len(cursors)+x[0][0])
        [intervals, lengths] = [list(z) for z in zip(*intervals)]
        return intervals, lengths

    @staticmethod
    def _compute_cursor_positions(length, n, start=0):
        d = length / n
        shifts = [int(start + d * j) for j in range(0, n)]
        return shifts

    @staticmethod
    def _shift_cursors(cursors, data_len, d=1):
        cs = list()
        for c in cursors:
            c += d
            while c >= data_len:
                c -= data_len
            while c < 0:
                c += data_len
            cs.append(c)
        return cs

    @staticmethod
    def _create_batch(data, cursors):
        return [data[c] for c in cursors]

    def _reset_cursors(self, data, batch_size, shuffle):
        data_len = len(data)
        if shuffle:
            cursors = self.random.sample(list(range(data_len)), batch_size)
        else:
            cursors = self._compute_cursor_positions(data_len, batch_size)
        cursors = sorted(cursors)
        if self._start_char is None:
            last_batch = self._create_batch(
                data,
                self._shift_cursors(cursors, data_len, d=-1)
            )
        else:
            last_batch = [self._start_char]*batch_size
        return cursors, last_batch

    def gen_batches(self, batch_size_and_num_unrollings: Union[int, List[int]], data_type: str = 'train',
                    shuffle: bool = None) -> Generator:
        """Return a generator, which generates (no preprocessing such as tokenization) batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
            shuffle (bool): whether to shuffle dataset before batching
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        if shuffle is None:
            shuffle = self.shuffle
        if isinstance(batch_size_and_num_unrollings, int):
            batch_size = batch_size_and_num_unrollings
            num_unrollings = 1
        else:
            batch_size = batch_size_and_num_unrollings[0]
            num_unrollings = batch_size_and_num_unrollings[1]

        data = self.data[data_type]
        cursors, last_batch = self._reset_cursors(data, batch_size, shuffle)
        data_len = len(data)

        if data_len == 0:
            return
        intervals, lengths = self._get_intervals_between_cursors(data_len, cursors)
        min_length = lengths[0]
        max_length = lengths[-1]
        num_batches = min_length // num_unrollings if self._no_intersections_in_epoch \
            else max_length // num_unrollings + 1
        # print("(OneStrLmIterator.gen_batches)num_batches:", num_batches)
        # print("(OneStrLmIterator.gen_batches)lengths:", lengths)
        num_missed = 0
        num_intersections = 0
        for length in lengths:
            num_missed += length - num_batches * num_unrollings
            num_intersections +=\
                num_batches * num_unrollings - length
        if self._verbose:
            if self._no_intersections_in_epoch:
                print('WARNING: %s characters will not be processed during this epoch!' % num_missed)
            else:
                print('WARNING: %s extra characters will be used during this epoch '
                      '(this means that number_of_batches*batch_size exceeds dataset size by %s)!'
                      % (num_intersections, num_intersections))
        for _ in range(num_batches):
            lb = list(last_batch)
            sequence = [lb]
            for _ in range(num_unrollings):
                b = self._create_batch(data, cursors)
                sequence.append(b)
                cursors = self._shift_cursors(cursors, data_len)
            last_batch = b
            yield sequence[:-1], sequence[1:]

    def get_instances(self, data_type: str = 'train') -> str:
        return [self.data[data_type]]

