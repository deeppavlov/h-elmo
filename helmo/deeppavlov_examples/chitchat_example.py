import deeppavlov
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.commands.train import build_model_from_config
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.models.tokenizers.lazy_tokenizer import LazyTokenizer
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.tf_model import TFModel


import json
import numpy as np
import tensorflow as tf

from itertools import chain
from pathlib import Path


download_decompress('http://files.deeppavlov.ai/datasets/personachat_v2.tar.gz', './personachat')

@register('personachat_dataset_reader')
class PersonaChatDatasetReader(DatasetReader):
    """
    PersonaChat dataset from
    Zhang S. et al. Personalizing Dialogue Agents: I have a dog, do you have pets too?
    https://arxiv.org/abs/1801.07243
    Also, this dataset is used in ConvAI2 http://convai.io/
    This class reads dataset to the following format:
    [{
        'persona': [list of persona sentences],
        'x': input utterance,
        'y': output utterance,
        'dialog_history': list of previous utterances
        'candidates': [list of candidate utterances]
        'y_idx': index of y utt in candidates list
      },
       ...
    ]
    """
    def read(self, dir_path: str, mode='self_original'):
        dir_path = Path(dir_path)
        dataset = {}
        for dt in ['train', 'valid', 'test']:
            dataset[dt] = self._parse_data(dir_path / '{}_{}.txt'.format(dt, mode))

        return dataset

    @staticmethod
    def _parse_data(filename):
        examples = []
        print(filename)
        curr_persona = []
        curr_dialog_history = []
        persona_done = False
        with filename.open('r') as fin:
            for line in fin:
                line = ' '.join(line.strip().split(' ')[1:])
                your_persona_pref = 'your persona: '
                if line[:len(your_persona_pref)] == your_persona_pref and persona_done:
                    curr_persona = [line[len(your_persona_pref):]]
                    curr_dialog_history = []
                    persona_done = False
                elif line[:len(your_persona_pref)] == your_persona_pref:
                    curr_persona.append(line[len(your_persona_pref):])
                else:
                    persona_done = True
                    x, y, _, candidates = line.split('\t')
                    candidates = candidates.split('|')
                    example = {
                        'persona': curr_persona,
                        'x': x,
                        'y': y,
                        'dialog_history': curr_dialog_history[:],
                        'candidates': candidates,
                        'y_idx': candidates.index(y)
                    }
                    curr_dialog_history.extend([x, y])
                    examples.append(example)

        return examples


data = PersonaChatDatasetReader().read('./personachat')

for k in data:
    print(k, len(data[k]))

data['train'][0]


@register('personachat_iterator')
class PersonaChatIterator(DataLearningIterator):
    def split(self, *args, **kwargs):
        for dt in ['train', 'valid', 'test']:
            setattr(self, dt, self._to_tuple(getattr(self, dt)))

    @staticmethod
    def _to_tuple(data):
        """
        Returns:
            list of (x, y)
        """
        return list(map(lambda x: (x['x'], x['y']), data))

iterator = PersonaChatIterator(data)
batch = [el for el in iterator.gen_batches(5, 'train')][0]
for x, y in zip(*batch):
    print('x:', x)
    print('y:', y)
    print('----------')


tokenizer = LazyTokenizer()
tokenizer(['Hello my friend'])


@register('dialog_vocab')
class DialogVocab(SimpleVocabulary):
    def fit(self, *args):
        tokens = chain(*args)
        super().fit(tokens)

    def __call__(self, batch, **kwargs):
        print("(DialogVocab.__call__)len(self):", len(self))
        indices_batch = []
        for utt in batch:
            print(utt)
            tokens = [self[token] for token in utt]
            indices_batch.append(tokens)
        return indices_batch


vocab = DialogVocab(
    save_path='./vocab.dict',
    load_path='./vocab.dict',
    min_freq=2,
    special_tokens=('<PAD>','<BOS>', '<EOS>', '<UNK>',),
    unk_token='<UNK>'
)

vocab.fit(tokenizer(iterator.get_instances(data_type='train')[0]), tokenizer(iterator.get_instances(data_type='train')[1]))
vocab.save()

vocab.freqs.most_common(10)

len(vocab)

vocab([['<BOS>', 'hello', 'my', 'friend', 'there_is_no_such_word_in_dataset', 'and_this', '<EOS>', '<PAD>']])

@register('sentence_padder')
class SentencePadder(Component):
    def __init__(self, length_limit, pad_token_id=0, start_token_id=1, end_token_id=2, *args, **kwargs):
        self.length_limit = length_limit
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

    def __call__(self, batch):
        for i in range(len(batch)):
            batch[i] = batch[i][:self.length_limit]
            batch[i] = [self.start_token_id] + batch[i] + [self.end_token_id]
            batch[i] += [self.pad_token_id] * (self.length_limit + 2 - len(batch[i]))
        return batch

padder = SentencePadder(length_limit=6)
vocab(padder(vocab([['hello', 'my', 'friend', 'there_is_no_such_word_in_dataset', 'and_this']])))


def encoder(inputs, inputs_len, embedding_matrix, cell_size, keep_prob=1.0):
    # inputs: tf.int32 tensor with shape bs x seq_len with token ids
    # inputs_len: tf.int32 tensor with shape bs
    # embedding_matrix: tf.float32 tensor with shape vocab_size x vocab_dim
    # cell_size: hidden size of recurrent cell
    # keep_prob: dropout keep probability
    with tf.variable_scope('encoder'):
        # first of all we should embed every token in input sequence (use tf.nn.embedding_lookup, don't forget about dropout)
        x_emb = tf.nn.dropout(tf.nn.embedding_lookup(embedding_matrix, inputs), keep_prob=keep_prob)

        # define recurrent cell (LSTM or GRU)
        encoder_cell = tf.nn.rnn_cell.GRUCell(
            num_units=cell_size,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='encoder_cell')

        # use tf.nn.dynamic_rnn to encode input sequence, use actual length of input sequence
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=x_emb, sequence_length=inputs_len,
                                                           dtype=tf.float32)
    return encoder_outputs, encoder_state

tf.reset_default_graph()
vocab_size = 100
hidden_dim = 100
inputs = tf.cast(tf.random_uniform(shape=[32, 10]) * vocab_size, tf.int32) # bs x seq_len
mask = tf.cast(tf.random_uniform(shape=[32, 10]) * 2, tf.int32) # bs x seq_len
inputs_len = tf.reduce_sum(mask, axis=1)
embedding_matrix = tf.random_uniform(shape=[vocab_size, hidden_dim])

encoder(inputs, inputs_len, embedding_matrix, hidden_dim)


def decoder(encoder_outputs, encoder_state, embedding_matrix, mask,
            cell_size, max_length, y_ph,
            start_token_id=1, keep_prob=1.0,
            teacher_forcing_rate_ph=None,
            use_attention=False, is_train=True):
    # decoder
    # encoder_outputs: tf.float32 tensor with shape bs x seq_len x encoder_cell_size
    # encoder_state: tf.float32 tensor with shape bs x encoder_cell_size
    # embedding_matrix: tf.float32 tensor with shape vocab_size x vocab_dim
    # mask: tf.int32 tensor with shape bs x seq_len with zeros for masked sequence elements
    # cell_size: hidden size of recurrent cell
    # max_length: max length of output sequence
    # start_token_id: id of <BOS> token in vocabulary
    # keep_prob: dropout keep probability
    # teacher_forcing_rate_ph: rate of using teacher forcing on each decoding step
    # use_attention: use attention on encoder outputs or use only encoder_state
    # is_train: is it training or inference? at inference time we can't use teacher forcing
    with tf.variable_scope('decoder'):
        # define decoder recurrent cell
        decoder_cell = tf.nn.rnn_cell.GRUCell(
            num_units=cell_size,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='decoder_cell')

        # initial value of output_token on previsous step is start_token
        output_token = tf.ones(shape=(tf.shape(encoder_outputs)[0],), dtype=tf.int32) * start_token_id
        # let's define initial value of decoder state with encoder_state
        decoder_state = encoder_state

        pred_tokens = []
        logits = []

        # use for loop to sequentially call recurrent cell
        for i in range(max_length):
            """
            TEACHER FORCING
            # here you can try to implement teacher forcing for your model
            # details about teacher forcing are explained further in tutorial

            # pseudo code:
            NOTE THAT FOLLOWING CONDITIONS SHOULD BE EVALUATED AT GRAPH RUNTIME
            use tf.cond and tf.logical operations instead of python if

            if i > 0 and is_train and random_value < teacher_forcing_rate_ph:
                input_token = y_ph[:, i-1]
            else:
                input_token = output_token

            input_token_emb = tf.nn.embedding_lookup(embedding_matrix, input_token)

            """
            if i > 0:
                input_token_emb = tf.cond(
                    tf.logical_and(
                        is_train,
                        tf.random_uniform(shape=(), maxval=1) <= teacher_forcing_rate_ph
                    ),
                    lambda: tf.nn.embedding_lookup(embedding_matrix, y_ph[:, i - 1]),  # teacher forcing
                    lambda: tf.nn.embedding_lookup(embedding_matrix, output_token)
                )
            else:
                input_token_emb = tf.nn.embedding_lookup(embedding_matrix, output_token)

            """
            ATTENTION MECHANISM
            # here you can add attention to your model
            # you can find details about attention further in tutorial
            """
            if use_attention:
                # compute attention and concat attention vector to input_token_emb
                att = dot_attention(encoder_outputs, decoder_state, mask, scope='att')
                input_token_emb = tf.concat([input_token_emb, att], axis=-1)

            input_token_emb = tf.nn.dropout(input_token_emb, keep_prob=keep_prob)
            # call recurrent cell
            decoder_outputs, decoder_state = decoder_cell(input_token_emb, decoder_state)
            decoder_outputs = tf.nn.dropout(decoder_outputs, keep_prob=keep_prob)
            # project decoder output to embeddings dimension
            embeddings_dim = embedding_matrix.get_shape()[1]
            output_proj = tf.layers.dense(decoder_outputs, embeddings_dim, activation=tf.nn.tanh,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='proj', reuse=tf.AUTO_REUSE)
            # compute logits
            output_logits = tf.matmul(output_proj, embedding_matrix, transpose_b=True)

            logits.append(output_logits)
            output_probs = tf.nn.softmax(output_logits)
            output_token = tf.argmax(output_probs, axis=-1)
            pred_tokens.append(output_token)

        y_pred_tokens = tf.transpose(tf.stack(pred_tokens, axis=0), [1, 0])
        y_logits = tf.transpose(tf.stack(logits, axis=0), [1, 0, 2])
    return y_pred_tokens, y_logits

tf.reset_default_graph()
vocab_size = 100
hidden_dim = 100
inputs = tf.cast(tf.random_uniform(shape=[32, 10]) * vocab_size, tf.int32) # bs x seq_len
mask = tf.cast(tf.random_uniform(shape=[32, 10]) * 2, tf.int32) # bs x seq_len
inputs_len = tf.reduce_sum(mask, axis=1)
embedding_matrix = tf.random_uniform(shape=[vocab_size, hidden_dim])

teacher_forcing_rate = tf.random_uniform(shape=())
y = tf.cast(tf.random_uniform(shape=[32, 10]) * vocab_size, tf.int32)

encoder_outputs, encoder_state = encoder(inputs, inputs_len, embedding_matrix, hidden_dim)
decoder(encoder_outputs, encoder_state, embedding_matrix, mask, hidden_dim, max_length=10,
        y_ph=y, teacher_forcing_rate_ph=teacher_forcing_rate)


@register('seq2seq')
class Seq2Seq(TFModel):
    def __init__(self, **kwargs):
        # hyperparameters

        # dimension of word embeddings
        self.embeddings_dim = kwargs.get('embeddings_dim', 100)
        # size of recurrent cell in encoder and decoder
        self.cell_size = kwargs.get('cell_size', 200)
        # dropout keep_probability
        self.keep_prob = kwargs.get('keep_prob', 0.8)
        # learning rate
        self.learning_rate = kwargs.get('learning_rate', 3e-04)
        # max length of output sequence
        self.max_length = kwargs.get('max_length', 20)
        self.grad_clip = kwargs.get('grad_clip', 5.0)
        self.start_token_id = kwargs.get('start_token_id', 1)
        self.vocab_size = kwargs.get('vocab_size', 11595)
        self.teacher_forcing_rate = kwargs.get('teacher_forcing_rate', 0.0)
        self.use_attention = kwargs.get('use_attention', False)

        # create tensorflow session to run computational graph in it
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self.init_graph()

        # define train op
        self.train_op = self.get_train_op(self.loss, self.lr_ph,
                                          optimizer=tf.train.AdamOptimizer,
                                          clip_norm=self.grad_clip)
        # initialize graph variables
        self.sess.run(tf.global_variables_initializer())

        super().__init__(**kwargs)
        # load saved model if there is one
        if self.load_path is not None:
            self.load()

    def init_graph(self):
        # create placeholders
        self.init_placeholders()

        self.x_mask = tf.cast(self.x_ph, tf.int32)
        self.y_mask = tf.cast(self.y_ph, tf.int32)

        self.x_len = tf.reduce_sum(self.x_mask, axis=1)

        # create embeddings matrix for tokens
        self.embeddings = tf.Variable(
            tf.random_uniform((self.vocab_size, self.embeddings_dim), -0.1, 0.1, name='embeddings'), dtype=tf.float32)

        # encoder
        encoder_outputs, encoder_state = encoder(self.x_ph, self.x_len, self.embeddings, self.cell_size,
                                                 self.keep_prob_ph)

        # decoder
        self.y_pred_tokens, y_logits = decoder(encoder_outputs, encoder_state, self.embeddings, self.x_mask,
                                               self.cell_size, self.max_length,
                                               self.y_ph, self.start_token_id, self.keep_prob_ph,
                                               self.teacher_forcing_rate_ph, self.use_attention, self.is_train_ph)

        # loss
        self.y_ohe = tf.one_hot(self.y_ph, depth=self.vocab_size)
        self.y_mask = tf.cast(self.y_mask, tf.float32)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_ohe, logits=y_logits) * self.y_mask
        self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(self.y_mask)

    def init_placeholders(self):
        # placeholders for inputs
        self.x_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='x_ph')
        # at inference time y_ph is used (y_ph exists in computational graph)  when teacher forcing is activated, so we add dummy default value
        # this dummy value is not actually used at inference
        self.y_ph = tf.placeholder_with_default(tf.zeros_like(self.x_ph), shape=(None, None), name='y_ph')

        # placeholders for model parameters
        self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name='lr_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')
        self.teacher_forcing_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='teacher_forcing_rate_ph')

    def _build_feed_dict(self, x, y=None):
        feed_dict = {
            self.x_ph: x,
        }
        if y is not None:
            feed_dict.update({
                self.y_ph: y,
                self.lr_ph: self.learning_rate,
                self.keep_prob_ph: self.keep_prob,
                self.is_train_ph: True,
                self.teacher_forcing_rate_ph: self.teacher_forcing_rate,
            })
        return feed_dict

    def train_on_batch(self, x, y):
        feed_dict = self._build_feed_dict(x, y)
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def __call__(self, x):
        feed_dict = self._build_feed_dict(x)
        y_pred = self.sess.run(self.y_pred_tokens, feed_dict=feed_dict)
        return y_pred

s2s = Seq2Seq(
    save_path='YOUR_PATH_TO_WORKING_DIR/model',
    load_path='YOUR_PATH_TO_WORKING_DIR/model'
)

vocab(s2s(padder(vocab([['hello', 'my', 'friend', 'there_is_no_such_word_in_dataset', 'and_this']]))))

def softmax_mask(values, mask):
    # adds big negative to masked values
    INF = 1e30
    return -INF * (1 - tf.cast(mask, tf.float32)) + values


def dot_attention(memory, state, mask, scope="dot_attention"):
    # inputs: bs x seq_len x hidden_dim
    # state: bs x hidden_dim
    # mask: bs x seq_len
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # dot product between each item in memory and state
        logits = tf.matmul(memory, tf.expand_dims(state, axis=1), transpose_b=True)
        logits = tf.squeeze(logits, [2])

        # apply mask to logits
        logits = softmax_mask(logits, mask)

        # apply softmax to logits
        att_weights = tf.expand_dims(tf.nn.softmax(logits), axis=2)

        # compute weighted sum of items in memory
        att = tf.reduce_sum(att_weights * memory, axis=1)
        return att


tf.reset_default_graph()
memory = tf.random_normal(shape=[32, 10, 100]) # bs x seq_len x hidden_dim
state = tf.random_normal(shape=[32, 100]) # bs x hidden_dim
mask = tf.cast(tf.random_normal(shape=[32, 10]), tf.int32) # bs x seq_len
dot_attention(memory, state, mask)


@register('postprocessing')
class SentencePostprocessor(Component):
    def __init__(self, pad_token='<PAD>', start_token='<BOS>', end_token='<EOS>', *args, **kwargs):
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token

    def __call__(self, batch):
        for i in range(len(batch)):
            batch[i] = ' '.join(self._postproc(batch[i]))
        return batch

    def _postproc(self, utt):
        if self.end_token in utt:
            utt = utt[:utt.index(self.end_token)]
        return utt

postprocess = SentencePostprocessor()
postprocess(vocab(s2s(padder(vocab([['hello', 'my', 'friend', 'there_is_no_such_word_in_dataset', 'and_this']])))))

config = {
  "dataset_reader": {
    "name": "personachat_dataset_reader",
    "data_path": "./personachat"
  },
  "dataset_iterator": {
    "name": "personachat_iterator",
    "seed": 1337,
    "shuffle": True
  },
  "chainer": {
    "in": ["x"],
    "in_y": ["y"],
    "pipe": [
      {
        "name": "lazy_tokenizer",
        "id": "tokenizer",
        "in": ["x"],
        "out": ["x_tokens"]
      },
      {
        "name": "lazy_tokenizer",
        "id": "tokenizer",
        "in": ["y"],
        "out": ["y_tokens"]
      },
      {
        "name": "dialog_vocab",
        "id": "vocab",
        "save_path": "./vocab.dict",
        "load_path": "./vocab.dict",
        "min_freq": 2,
        "special_tokens": ["<PAD>","<BOS>", "<EOS>", "<UNK>"],
        "unk_token": "<UNK>",
        "fit_on": ["x_tokens", "y_tokens"],
        "in": ["x_tokens"],
        "out": ["x_tokens_ids"]
      },
      {
        "ref": "vocab",
        "in": ["y_tokens"],
        "out": ["y_tokens_ids"]
      },
      {
        "name": "sentence_padder",
        "id": "padder",
        "length_limit": 20,
        "in": ["x_tokens_ids"],
        "out": ["x_tokens_ids"]
      },
      {
        "ref": "padder",
        "in": ["y_tokens_ids"],
        "out": ["y_tokens_ids"]
      },
      {
        "name": "seq2seq",
        "id": "s2s",
        "max_length": "#padder.length_limit+2",
        "cell_size": 250,
        "embeddings_dim": 50,
        "vocab_size": 11595,
        "keep_prob": 0.8,
        "learning_rate": 3e-04,
        "teacher_forcing_rate": 0.0,
        "use_attention": False,
        "save_path": "YOUR_PATH_TO_WORKING_DIR/model",
        "load_path": "YOUR_PATH_TO_WORKING_DIR/model",
        "in": ["x_tokens_ids"],
        "in_y": ["y_tokens_ids"],
        "out": ["y_predicted_tokens_ids"],
      },
      {
        "ref": "vocab",
        "in": ["y_predicted_tokens_ids"],
        "out": ["y_predicted_tokens"]
      },
      {
        "name": "postprocessing",
        "in": ["y_predicted_tokens"],
        "out": ["y_predicted_tokens"]
      }
    ],
    "out": ["y_predicted_tokens"]
  },
  "train": {
    "log_every_n_batches": 100,
    "val_every_n_epochs":0,
    "batch_size": 64,
    "validation_patience": 0,
    "epochs": 20,
    "metrics": ["bleu"],
  }
}

print("Before building from config")
model = build_model_from_config(config)
model(['Hi, how are you?', 'Any ideas my dear friend?'])
json.dump(config, open('seq2seq.json', 'w'))
train_evaluate_model_from_config('seq2seq.json')

model = build_model_from_config(config)
model(['hi, how are you?', 'any ideas my dear friend?', 'okay, i agree with you', 'good bye!'])