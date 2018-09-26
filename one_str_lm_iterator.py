from random import Random
from typing import Dict, Tuple, List, Generator, Any, Union

import numpy as np

# from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
# from deeppavlov.core.common.registry import register

import sys
from util.deal_with_cephfs import add_repo_2_sys_path
sys.path = add_repo_2_sys_path('DeepPavlov')

from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


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
                 num_unrollings: int = 1,
                 seed: int = None,
                 shuffle: bool = False,
                 shuffle_amplitude: int = 10,
                 start_char: str = None,
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
        self.num_unrollings = num_unrollings

        self.shuffle = shuffle
        self._default_shuffle_amplitude = shuffle_amplitude

        self.random = Random(seed)

        self._start_char = start_char

        self.train = data.get('train', "")
        self.valid = data.get('valid', "")
        self.test = data.get('test', "")
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }
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
        sorted_cursors = sorted(cursors)
        intervals_and_lengths = list()
        for idx in range(len(cursors)-1):
            intervals_and_lengths.append(
                self._compute_interval(sorted_cursors[idx], sorted_cursors[idx+1], data_len)
            )
        intervals_and_lengths.append(
            (
                self._compute_interval(sorted_cursors[-1], sorted_cursors[0], data_len, loop=True)
            )
        )
        intervals_and_lengths = sorted(
            intervals_and_lengths,
            key=lambda x: cursors.index(x[0][0])*data_len + x[1],
            reverse=True,
        )
        [intervals, lengths] = [list(z) for z in zip(*intervals_and_lengths)]
        return intervals, lengths

    @staticmethod
    def _compute_cursor_positions(length, n, start=0):
        d = length // n
        shifts = [start + d * j for j in range(0, n)]
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

    def _shuffle_cursors(self, cursors, amplitude, data_len):
        _, lengths = self._get_intervals_between_cursors(data_len, cursors)
        min_length = min(lengths)
        if amplitude > min_length // 2 + 1:
            amplitude = min_length // 2 + 1
        shifts = self.random.choices(list(range(-int(amplitude), int(amplitude))), k=len(cursors))
        cursors = [c + s for c, s in zip(cursors, shifts)]
        projected_cursors = list()
        for c in cursors:
            while c < 0:
                c += data_len
            while c > data_len:
                c -= data_len
            projected_cursors.append(c)
        intervals, lengths = self._get_intervals_between_cursors(data_len, projected_cursors)
        [intervals, _] = [list(z) for z in zip(*sorted(zip(intervals, lengths), key=lambda x: x[1]))]
        return [inter[0] for inter in intervals]

    @staticmethod
    def _create_batch(data, cursors):
        return [data[c] for c in cursors]

    def reset_cursors(self, data, batch_size, shuffle, shuffle_amplitude=None):
        if shuffle_amplitude is None:
            shuffle_amplitude = self._default_shuffle_amplitude
        data_len = len(data)
        cursors = self._compute_cursor_positions(data_len, batch_size)
        if shuffle:
            cursors = self._shuffle_cursors(cursors, shuffle_amplitude, data_len)
        if self._start_char is None:
            last_batch = self._create_batch(
                data,
                cursors,
            )
        else:
            last_batch = [self._start_char]*batch_size
        return cursors, last_batch

    @staticmethod
    def _transform_output(output):
        return list(np.array(output).transpose())

    @staticmethod
    def _transform_unplanned_batches(unplanned_batches):
        new_unplanned_batches = list()
        max_batch_size = 0
        max_num_unrollings = 0
        for k, v in unplanned_batches.items():
            new_unplanned_batches.append(
                dict(
                    [('batch size', k[0]), ('num unrollings', k[1]), ('num occurrences', v)]
                )
            )
            if k[0] > max_batch_size:
                max_batch_size = k[0]
            if k[1] > max_num_unrollings:
                max_num_unrollings = k[1]
        return sorted(
            new_unplanned_batches,
            key=lambda x: x['batch size']*max_num_unrollings + x['num unrollings'],
            reverse=True
        )

    def gen_batches(self, batch_size: Union[int, List[int]], data_type: str = 'train',
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

        data = self.data[data_type]
        cursors, last_batch = self.reset_cursors(data, batch_size, shuffle)
        data_len = len(data)

        if data_len == 0:
            return
        _, lengths = self._get_intervals_between_cursors(data_len, cursors)
        min_length = min(lengths)
        max_length = max(lengths)
        num_batches_with_initial_batch_specs = min_length // self.num_unrollings if self._start_char is None else \
            (min_length + 1) // self.num_unrollings

        # print("(OneStrLmIterator.gen_batches)lengths:", lengths)
        # print(
        #     "(OneStrLmIterator.gen_batches)num_batches_with_initial_batch_specs:",num_batches_with_initial_batch_specs)
        for batch_idx in range(num_batches_with_initial_batch_specs):
            lb = list(last_batch)
            sequence = [lb]
            for _ in range(self.num_unrollings):
                cursors = self._shift_cursors(cursors, data_len)
                b = self._create_batch(data, cursors)
                sequence.append(b)
            last_batch = b
            yield self._transform_output(sequence[:-1]), self._transform_output(sequence[1:])

        lengths = np.array(lengths)
        num_batches_done = num_batches_with_initial_batch_specs
        char_counters = np.array([num_batches_done*self.num_unrollings]*batch_size) if self._start_char is None else \
            np.array([num_batches_done * self.num_unrollings - 1] * batch_size)
        unplanned_batches = dict()

        if np.any(char_counters >= lengths):
            cursors = list(np.array(cursors)[char_counters < lengths])
            char_counters, lengths = char_counters[char_counters < lengths], lengths[char_counters < lengths]
        sequence = [self._create_batch(data, cursors)]
        current_batch_size = len(cursors)
        current_num_unrollings = 0

        while np.any(char_counters < max_length):
            # print(
            #     "after while(OneStrLmIterator.gen_batches)char_counters:",
            #     char_counters)
            # print(
            #     "after while(OneStrLmIterator.gen_batches)lengths:",
            #     lengths)
            # print(
            #     "after while(OneStrLmIterator.gen_batches)sequence:",
            #     sequence)
            cursors = self._shift_cursors(cursors, data_len)
            sequence.append(self._create_batch(data, cursors))
            char_counters += 1
            current_num_unrollings += 1

            if np.any(char_counters >= lengths) or current_num_unrollings >= self.num_unrollings:
                # print(
                #     "(OneStrLmIterator.gen_batches)char_counters:",
                #     char_counters)
                # print(
                #     "(OneStrLmIterator.gen_batches)lengths:",
                #     lengths)
                # print(
                #     "(OneStrLmIterator.gen_batches)sequence:",
                #     sequence)
                yield self._transform_output(sequence[:-1]), self._transform_output(sequence[1:])
                num_batches_done += 1
                if current_batch_size != batch_size or current_num_unrollings != self.num_unrollings:
                    specs = (current_batch_size, current_num_unrollings)
                    if specs in unplanned_batches:
                        unplanned_batches[specs] += 1
                    else:
                        unplanned_batches[specs] = 1

                cursors = list(np.array(cursors)[char_counters < lengths])
                char_counters, lengths = char_counters[char_counters < lengths], lengths[char_counters < lengths]
                sequence = [self._create_batch(data, cursors)]
                current_batch_size = len(cursors)
                current_num_unrollings = 0

        unplanned_batches = self._transform_unplanned_batches(unplanned_batches)
        if self._verbose:
            log.info("total_batches in epoch on dataset '{}': {}".format(data_type, num_batches_done))
            log.info(
                "batches with unplanned sizes or unplanned num unrollings on dataset {}:\n{}".format(
                    data_type, str(unplanned_batches)))







    def get_instances(self, data_type: str = 'train') -> str:
        return [self.data[data_type]]
