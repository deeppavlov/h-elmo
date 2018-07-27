from random import Random
from typing import Dict, Tuple, List, Generator, Any
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.common.registry import register


@register('str_lm_iterator')
class StrLmIterator(DataLearningIterator):

    def __init__(self, data: Dict[str, List[Tuple[Any, Any]]],
                 seed: int = None, shuffle: bool = True, start_char: str = '\n',
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
        self._cursors = dict(
            train=list(),
            valid=list(),
            test=list(),
        )
        self._last_batches = dict(
            train=None,
            valid=None,
            test=None,
        )
        self._old_batch_size = None

    @staticmethod
    def _get_intervals_between_cursors(data_len, cursors):
        intervals = list()
        for idx in range(len(cursors)-1):
            intervals.append(
                (
                    (cursors[idx], cursors[idx+1]),
                    cursors[idx + 1] - cursors[idx]
                )
            )
        intervals.append(
            (
                (data_len-cursors[-1]+cursors[0]),
                data_len - cursors[-1] + cursors[0]
            )
        )
        intervals = sorted(intervals, key=lambda x: x[1]*len(cursors)+x[0][0])
        [intervals, lengths] = [list(z) for z in zip(*intervals)]
        return intervals, lengths

    @staticmethod
    def _compute_shifts(length, n):
        d = length / n
        shifts = [int(d * j) for j in range(1, n + 1)]
        return shifts

    def _get_new_cursor_shifts_min_len_const(self, lengths, num_inserts):
        min_len = lengths[-1]
        shifts = [None] * len(lengths)
        num_made_inserts = 0
        i = 0
        while num_made_inserts < num_inserts:
            l = lengths[i]
            n = l // min_len - 1
            shifts[i] = self._compute_shifts(l, n)
            i += 1
            num_made_inserts += n
        return shifts

    def _get_new_cursor_shifts_min_len_change(self, lengths, num_inserts):
        shifts = [None] * len(lengths)
        interval_lengths_after_insert = [l // 2 for l in lengths]
        n_ins_on_inter = [0] * len(lengths)
        for i in range(num_inserts):
            idx = interval_lengths_after_insert.index(max(interval_lengths_after_insert))
            n_ins_on_inter[idx] += 1
            interval_lengths_after_insert[idx] = lengths[idx] // (n_ins_on_inter[idx] + 2)
        for i, n_ins in enumerate(n_ins_on_inter):
            shifts[i] = self._compute_shifts(lengths[i], n_ins)
        return shifts

    def _get_new_cursor_shifts_by_interval(self, lengths, num_inserts):
        min_len = lengths[-1]
        num_easy_inserts = 0
        idx = 0
        while lengths[idx] // min_len > 1 and idx < len(lengths) - 1 and num_easy_inserts < num_inserts:
            num_easy_inserts += lengths[idx] // min_len - 1
            idx += 1
        if num_easy_inserts >= num_inserts:
            shifts = self._get_new_cursor_shifts_min_len_const(lengths, num_inserts)
        else:
            shifts = self._get_new_cursor_shifts_min_len_change(lengths, num_inserts)
        return shifts

    def _change_batch_size(self, new_batch_size, cursors, last_batches, data_len):
        if new_batch_size < self._old_batch_size:
            new_cursors = cursors[:new_batch_size]
            new_last_batches = last_batches[:new_batch_size]
        elif new_batch_size > self._old_batch_size:
            intervals, lengths = self._get_intervals_between_cursors(data_len, cursors)
            num_inserts = self._get_new_cursor_shifts_by_interval(lengths, new_batch_size-self._old_batch_size)

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None, reset: bool = None) -> Generator:
        """Return a generator, which serves for generation of raw (no preprocessing such as tokenization)
        batches
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
        cursors = self._cursors[data_type]
        last_batches = self._last_batches[data_type]
        data_len = len(data)

        if data_len == 0:
            return
        if batch_size != self._old_batch_size:
            pass

        order = list(range(data_len))
        if shuffle:
            self.random.shuffle(order)

        if batch_size < 0:
            batch_size = data_len

        for i in range((data_len - 1) // batch_size + 1):
            yield tuple(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))