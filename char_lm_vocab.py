from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.data.simple_vocab import SimpleVocabulary

log = get_logger(__name__)


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
        indices_batch = []
        for token in batch:
            indices_batch.append(self[token])
        return indices_batch

    def save(self):
        log.info("[saving vocabulary to {}]".format(self.save_path))
        with self.save_path.open('wt') as f:
            for n in range(len(self)):
                token = self._i2t[n]
                cnt = self.freqs[token]
                f.write('{}\t{:d}\n'.format(token, cnt))

    def load(self):
        self.reset()
        if self.load_path:
            if self.load_path.is_file():
                log.info("[loading vocabulary from {}]".format(self.load_path))
                tokens, counts = [], []
                for ln in self.load_path.open('r'):
                    token, cnt = ln.split('\t', 1)
                    tokens.append(token)
                    counts.append(int(cnt))
                self._add_tokens_with_freqs(tokens, counts)
            elif isinstance(self.load_path, Path):
                if not self.load_path.parent.is_dir():
                    raise ConfigError("Provided `load_path` for {} doesn't exist!".format(
                        self.__class__.__name__))
        else:
            raise ConfigError("`load_path` for {} is not provided!".format(self))

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
