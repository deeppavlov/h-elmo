from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
import codecs

import numpy as np
np.set_printoptions(threshold=np.nan)

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.data.simple_vocab import SimpleVocabulary

log = get_logger(__name__)


def apply_func_on_depth(obj, func, depth, permeable_types=(list, tuple, dict)):
    if depth != 0 and isinstance(obj, permeable_types):
        if isinstance(obj, (list, tuple)):
            processed = list()
            for elem in obj:
                processed.append(apply_func_on_depth(elem, func, depth-1, permeable_types=permeable_types))
            if isinstance(obj, tuple):
                processed = tuple(processed)
            return processed
        elif isinstance(obj, dict):
            processed = dict()
            for key, value in obj.items():
                processed[key] = apply_func_on_depth(value, depth-1, permeable_types=permeable_types)
            return processed
    return func(obj)


@register('char_lm_vocab')
class CharLMVocabulary(SimpleVocabulary):
    def __init__(self,
                 special_tokens=tuple(),
                 max_tokens=2**30,
                 pad_with_zeros=False,
                 unk_token=None,
                 *args,
                 **kwargs):
        super().__init__(**kwargs)
        self.special_tokens = special_tokens
        self._max_tokens = max_tokens
        self._min_freq = 1
        self._pad_with_zeros = pad_with_zeros
        self.unk_token = unk_token
        self.reset()
        if self.load_path:
            self.load()

    def fit(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        self.reset()
        self.freqs = Counter(chain(*[list(text) for text in texts]))
        for token, freq in self.freqs.most_common()[:self._max_tokens]:
            if freq >= self._min_freq:
                self._t2i[token] = self.count
                self._i2t.append(token)
                self.count += 1

    def _add_tokens_with_freqs(self, tokens, freqs):
        self.freqs = Counter()
        self.freqs.update(dict(zip(tokens, freqs)))
        for token, freq in zip(tokens, freqs):
            if freq >= self._min_freq:
                self._t2i[token] = self.count
                self._i2t.append(token)
                self.count += 1

    def __call__(self, batch, **kwargs):
        batch = np.array(batch)
        if 'str' in batch.dtype.name:
            f = np.vectorize(lambda x: self[x])
            indices_batch = f(batch)

        else:
            # print(np.argmax(batch, axis=-1))
            # print(len(self))
            indices_batch = np.apply_along_axis(lambda x: self[np.argmax(x)], -1, batch)
        if any([i.dtype.name == 'object' for i in indices_batch]):
            print("(CharLMVocabulary.__call__)len(self):", len(self))
            print("(CharLMVocabulary.__call__)self['\n']:", self['\n'])
            print("(CharLMVocabulary.__call__)self._t2i:", self._t2i)
            g = np.vectorize(lambda x: x is None)
            mask = g(indices_batch)
            print("(CharLMVocabulary.__call__)batch[mask]:", batch[mask])
            print("(CharLMVocabulary.__call__)indices_batch[0]:", indices_batch[0])
        return indices_batch

    def save(self):
        log.info("[saving vocabulary to {}]".format(self.save_path))
        with self.save_path.open('wt') as f:
            for n in range(len(self)):
                token = self._i2t[n]
                cnt = self.freqs[token]
                f.write('{}\t{:d}\n'.format(repr(token), cnt))

    def load(self):
        self.reset()
        if self.load_path:
            if self.load_path.is_file():
                log.info("[loading vocabulary from {}]".format(self.load_path))
                tokens, counts = [], []
                for ln in self.load_path.open('r'):
                    token, cnt = ln.split('\t', 1)
                    token = token[1:-1]
                    token = codecs.escape_decode(bytes(token, "utf-8"))[0].decode("utf-8")
                    tokens.append(token)
                    counts.append(int(cnt))
                self._add_tokens_with_freqs(tokens, counts)
            elif isinstance(self.load_path, Path):
                if not self.load_path.parent.is_dir():
                    raise ConfigError("Provided `load_path` for {} doesn't exist!".format(
                        self.__class__.__name__))
        else:
            raise ConfigError("`load_path` for {} is not provided!".format(self))
        print("(CharLMVocabulary.__call__)self._t2i:", self._t2i)

    def is_str_batch(self, batch):
        if not self.is_empty(batch):
            return True
        else:
            return False

    def reset(self):
        self.freqs = None
        self._t2i = defaultdict(lambda: None)
        self._i2t = []
        self.count = 0

    @staticmethod
    def is_empty(batch):
        non_empty = [item for item in batch if len(item) > 0]
        return len(non_empty) == 0
