import sys
from copy import deepcopy
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnLSTM as CudnnLSTM
from tensorflow.contrib.cudnn_rnn import CudnnGRU as CudnnGRU

from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary, char2vec, pred2vec, \
    pred2vec_fast, vec2char, vec2char_fast, char2id, id2char, get_available_gpus, device_name_scope, \
    average_gradients, InvalidArgumentError, \
    compose_save_list, compose_reset_list, compose_randomize_list, construct_dict_without_none_entries, \
    append_to_nested, get_average_with_weights_func, func_on_list_in_nested, tensor_stats
from learning_to_learn.pupils.pupil import Pupil

from learning_to_learn.tensors import compute_metrics_raw_lbls

import helmo.util.tensor_ops as tensor_ops


class LmFastBatchGenerator(object):
    @staticmethod
    def create_vocabulary(texts):
        text = ''
        for t in texts:
            text += t
        return create_vocabulary(text)

    @staticmethod
    def char2vec(char, character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return np.array([char2id(char, character_positions_in_vocabulary)])

    @staticmethod
    def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
        return np.reshape(pred2vec_fast(pred), (1, -1, 1))

    @staticmethod
    def vec2char(vec, vocabulary):
        # print("(LmFastBatchGenerator.vec2char)vec:", vec, file=sys.stderr)
        return vec2char(vec, vocabulary)

    @staticmethod
    def vec2char_fast(vec, vocabulary):
        return vec2char_fast(vec, vocabulary)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None, random_batch_initiation=False):
        # print("(resrnn.LmFastBatchGenerator)num_unrollings:", num_unrollings)
        # print("(resrnn.LmFastBatchGenerator)random_batch_initiation:", random_batch_initiation)
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabulary = vocabulary
        self._vocabulary_size = len(self.vocabulary)
        self.character_positions_in_vocabulary = get_positions_in_vocabulary(self.vocabulary)
        self._num_unrollings = num_unrollings
        if random_batch_initiation:
            self._cursor = random.sample(range(self._text_size), batch_size)
        else:
            segment = self._text_size // batch_size
            self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()
        # print("(resrnn.LmFastBatchGenerator.__init__)num batches in epoch:",
        #       len(self._text) // (self._num_unrollings * self._batch_size))

    def get_num_batches(self):
        # print(len(self._text) // (self._batch_size * self._num_unrollings))
        return len(self._text) // (self._batch_size * self._num_unrollings)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        return np.array([0 for _ in range(self._batch_size)])

    def _zero_batch(self):
        return -np.ones(shape=(self._batch_size), dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        ret = np.array([char2id(self._text[self._cursor[b]], self.character_positions_in_vocabulary)
                        for b in range(self._batch_size)])
        # if len(self._text) == 640000:
        #     # print("(resrnn.LmFastBatchGenerator._next_batch)length == 640000")
        #     ret = np.array([char2id(self._text[self._cursor[0]], self.character_positions_in_vocabulary)
        #                     for b in range(self._batch_size)])
        # else:
        #     ret = np.array([char2id(self._text[self._cursor[b]], self.character_positions_in_vocabulary)
        #                     for b in range(self._batch_size)])
        for b in range(self._batch_size):
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return ret

    def char2batch(self, char):
        return np.stack(char2vec(char, self.character_positions_in_vocabulary)), np.stack(self._zero_batch())

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        inps = np.stack(batches[:-1])
        lbls = np.stack(batches[1:])
        # print('(LmFastBatchGenerator.next)inps.shape:', inps.shape)
        # print('(LmFastBatchGenerator.next)lbls.shape:', lbls.shape)
        return inps, lbls


def characters(probabilities, vocabulary):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c, vocabulary) for c in np.argmax(probabilities, 1)]


def batches2string(batches, vocabulary):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [u""] * batches[0].shape[0]
    for b in batches:
        s = [u"".join(x) for x in zip(s, characters(b, vocabulary))]
    return s


class Rnn(Pupil):
    _name = 'rnn'
    _input_format = 'indices'

    @classmethod
    def check_kwargs(cls, **kwargs):
        pass

    @staticmethod
    def get_special_args():
        return dict()

    def _output_module(self, inp):
        with tf.name_scope('output_module'):
            for idx, out_core in enumerate(self._out_vars):
                inp = tf.einsum('ijk,kl->ijl', inp, out_core['matrix']) + out_core['bias']
                if idx < len(self._out_vars) - 1:
                    inp = tf.nn.relu(inp)
        return inp

    def _compute_output_matrix_parameters(self, idx):
        if idx == 0:
            # print('self._num_nodes:', self._num_nodes)
            input_dim = self._rnn_map['num_nodes'][-1]
        else:
            input_dim = self._num_out_nodes[idx - 1]
        if idx == self._num_out_layers - 1:
            output_dim = self._voc_size
        else:
            output_dim = self._num_out_nodes[idx]
        stddev = self._init_parameter * np.sqrt(1. / (input_dim + output_dim))
        return input_dim, output_dim, stddev

    def _l2_loss(self, matrices):
        with tf.name_scope('l2_loss'):
            regularizer = tf.contrib.layers.l2_regularizer(self._reg_rate)
            loss = 0
            for matr in matrices:
                loss += regularizer(matr)
            return loss

    def _select_optimizer(self):
        if self._optimizer_type == 'adam':
            self._optimizer = tf.train.AdamOptimizer(
                learning_rate=self._train_phds['learning_rate'])
        elif self._optimizer_type == 'rmsprop':
            self._optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self._train_phds['learning_rate'],
                decay=self._decay,
            )
        elif self._optimizer_type == 'adagrad':
            self._optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._train_phds['learning_rate'])
        elif self._optimizer_type == 'adadelta':
            self._optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=1.,
                rho=self._rho,
            )
        elif self._optimizer_type == 'momentum':
            self._optimizer = tf.train.MomentumOptimizer(
                learning_rate=self._train_phds['learning_rate'],
                momentum=self._train_phds['momentum']
            )
        elif self._optimizer_type == 'nesterov':
            self._optimizer = tf.train.MomentumOptimizer(
                learning_rate=self._train_phds['learning_rate'],
                momentum=self._train_phds['momentum'],
                use_nesterov=True
            )
        else:
            print('using sgd optimizer', file=sys.stderr)
            self._optimizer = tf.train.GradientDescentOptimizer(
                self._train_phds['learning_rate'])

    def _get_bs_by_gpu(self):
        batch_size = tf.shape(self._inp_and_lbl_phds['inps'])[1]
        bs_on_1st_gpu = tf.maximum(batch_size // self._num_gpus, 1)
        bs_on_last_gpu = tf.maximum(batch_size - (self._num_gpus - 1) * bs_on_1st_gpu, 0)
        return [bs_on_1st_gpu] * (self._num_gpus - 1) + [bs_on_last_gpu]

    def _add_embeding_graph(self, inps_by_gpu):
        outputs = list()
        with tf.name_scope('embed'):
            for inp, gpu_name in zip(inps_by_gpu, self._gpu_names):
                with tf.device(gpu_name), tf.name_scope(device_name_scope(gpu_name)):
                    x = tf.one_hot(inp, self._voc_size, dtype=tf.float32)
                    if self._embed_inputs:
                        outputs.append(
                            tf.einsum(
                                'ijk,kl->ijl',
                                x,
                                self._emb_vars['matrix']
                            ) + self._emb_vars['bias']
                        )
                    else:
                        outputs.append(x)
        return outputs

    def _adjust_dim(self, inp, target, matrix=None, bias=None):
        # print("(Rnn._adjust_dim)inp:", inp)
        # print(inp.get_shape())
        # print(inp.get_shape().as_list(), end='\n'*2)
        # print("(Rnn._adjust_dim)target:", target)
        # print(target.get_shape())
        # print(target.get_shape().as_list(), end='\n'*2)
        # print("(Rnn._adjust_dim)matrix:", matrix)
        # print(matrix.get_shape())
        # print(matrix.get_shape().as_list())
        # print('***************')
        if self._matrix_dim_adjustment:
            return tf.einsum('ijk,kl->ijl', inp, matrix) + bias
        else:
            with tf.name_scope('adjust_dim'):
                return tf.cond(
                    tf.shape(inp)[-1] > tf.shape(target)[-1],
                    true_fn=lambda: tensor_ops.reduce_last_dim(inp, target),
                    false_fn=lambda: tensor_ops.increase_last_dim(inp, target)
                )

    def _add_saved_state(self, rnn_map, gpu_name, state_name):
        if state_name not in rnn_map:
            rnn_map[state_name] = dict()
        rnn_map[state_name][gpu_name] = tensor_ops.get_saved_state_vars(
            len(rnn_map['num_nodes']), self._network_type, rnn_map['module_name'])
        if 'derived_branches' in rnn_map:
            for branch in rnn_map['derived_branches']:
                self._add_saved_state(branch, gpu_name, state_name)

    def _get_state_from_rnn_map(self, rnn_map, key, gpu_name):
        res = list()
        res.append(rnn_map[key][gpu_name])
        if 'derived_branches' in rnn_map:
            for branch in rnn_map['derived_branches']:
                res.extend(self._get_state_from_rnn_map(branch, key, gpu_name))
        return res

    def _prepare_init_state_resback(self, state_list, inp, rnn_map, gpu_name, saved_state_name, new_state_name):
        if 'derived_branches' in rnn_map:
            rnn_map['derived_branches'] = sorted(rnn_map['derived_branches'], key=lambda x: x['output_idx'])
        if new_state_name not in rnn_map:
            rnn_map[new_state_name] = dict()
        with tf.name_scope(rnn_map['module_name']):
            state_list.append(
                tensor_ops.prepare_init_state(
                    rnn_map[saved_state_name][gpu_name], inp, rnn_map['rnns'], self._network_type)
            )
            rnn_map[new_state_name][gpu_name] = len(state_list) - 1
            if 'derived_branches' in rnn_map:
                for branch in rnn_map['derived_branches']:
                    self._prepare_init_state_resback(
                        state_list, inp, branch, gpu_name, saved_state_name, new_state_name)

    def _add_states_from_derived_branches(self, rnn_idx, rnn_map, x, state, new_state_name, gpu_name):
        with tf.name_scope('collect_derived_branches_states'):
            if 'derived_branches' not in rnn_map:
                return x
            for branch in rnn_map['derived_branches']:
                if branch['input_idx'] == rnn_idx:
                    s = state[branch[new_state_name][gpu_name]][0] if self._rnn_type == 'gru' \
                        else state[branch[new_state_name][gpu_name]][0][0]
                    am = branch['in_back_adapter_matrix'] if 'in_back_adapter_matrix' in branch else None
                    ab = branch['in_back_adapter_bias'] if 'in_back_adapter_bias' in branch else None
                    x += self._adjust_dim(s, x, matrix=am, bias=ab)
        return x

    def _distribute_states(self, state, rnn_map, new_state_name, gpu_name):
        # print("(Rnn._distribute_states)rnn_map:", rnn_map)
        # print("(Rnn._distribute_states)new_state_name:", new_state_name)
        # print("(Rnn._distribute_states)rnn_map[new_state_name]:", rnn_map[new_state_name])
        # print("(Rnn._distribute_states)gpu_name:", gpu_name)
        rnn_map[new_state_name][gpu_name] = state[rnn_map[new_state_name][gpu_name]]
        if 'derived_branches' in rnn_map:
            for branch in rnn_map['derived_branches']:
                self._distribute_states(state, branch, new_state_name, gpu_name)

    def _rec_back_rnn_graph(
            self, x, state, rnn_map, gpu_name, training,
            saved_state_name, new_state_name, back_state
    ):
        branch_idx = 0
        intermediate = list()
        branch_length = len(rnn_map['rnns'])
        state_list = state[rnn_map[new_state_name][gpu_name]]
        # print("(Rnn._rec_back_rnn_graph)rnn_map:", rnn_map)
        # print("(Rnn._rec_back_rnn_graph)state_list:", state_list)
        # print("(Rnn._rec_back_rnn_graph)state:", state)
        # print(state)
        with tf.name_scope(rnn_map['module_name']):
            for rnn_idx, (rnn, s) in enumerate(
                    zip(
                        rnn_map['rnns'],
                        state_list,
                    )
            ):
                with tf.name_scope('apply_rnn_{}'.format(rnn_idx)):
                    if 'derived_branches' in rnn_map \
                            and branch_idx < len(rnn_map['derived_branches']) \
                            and rnn_map['derived_branches'][branch_idx]['output_idx'] == rnn_idx:
                        branch = rnn_map['derived_branches'][branch_idx]
                        branch_res = self._rec_back_rnn_graph(
                            intermediate[rnn_map['derived_branches'][branch_idx]['input_idx']], state,
                            branch, gpu_name, training,
                            saved_state_name, new_state_name,
                            state_list[rnn_map['derived_branches'][branch_idx]['output_idx']],
                        )
                        am = branch['adapter_matrix'] if 'adapter_matrix' in branch else None
                        ab = branch['adapter_bias'] if 'adapter_bias' in branch else None
                        x += self._adjust_dim(branch_res, x, matrix=am, bias=ab,)
                        branch_idx += 1
                    with tf.name_scope('add_backward_connection'):
                        if rnn_idx == branch_length - 1:
                            if back_state is not None:
                                st = back_state if self._rnn_type == 'gru' else back_state[0]
                                am = rnn_map['out_back_adapter_matrix'] if 'out_back_adapter_matrix' in rnn_map else None
                                ab = rnn_map['out_back_adapter_bias'] if 'out_back_adapter_bias' in rnn_map else None
                                x += self._adjust_dim(st, x, matrix=am, bias=ab,)
                        else:
                            st = state_list[rnn_idx + 1] if self._rnn_type == 'gru' else state_list[rnn_idx + 1][0]
                            am = rnn_map['intermediate_back_adapter_matrices'][rnn_idx] \
                                if 'intermediate_back_adapter_matrices' in rnn_map else None
                            ab = rnn_map['intermediate_back_adapter_biases'][rnn_idx] \
                                if 'intermediate_back_adapter_biases' in rnn_map else None
                            x += self._adjust_dim(st, x, matrix=am, bias=ab,)
                    x = self._add_states_from_derived_branches(rnn_idx, rnn_map, x, state, new_state_name, gpu_name)
                    # print("(Rnn._rec_back_rnn_graph)rnn.num_units:", rnn.num_units)
                    # print("(Rnn._rec_back_rnn_graph)x:", x)
                    # print("(Rnn._rec_back_rnn_graph)s:", s)
                    x, new_s = rnn(x, initial_state=s, training=training)
                    if rnn_map['input_idx'] is not None or rnn_idx < len(rnn_map['rnns']) - 1:
                        x = tf.nn.dropout(x, keep_prob=1. - self._reg_placeholders['dropout_rate'])
                    intermediate.append(x)
                    state_list[rnn_idx] = new_s
        return x

    def _build_rnns(self, inp_shape, rnn_map,):
        if 'derived_branches' in rnn_map:
            rnn_map['derived_branches'] = sorted(rnn_map['derived_branches'], key=lambda x: x['output_idx'])
        branch_idx = 0
        intermediate = list()
        branch_length = len(rnn_map['rnns'])
        with tf.variable_scope('build_rnn_vars'):
            for rnn_idx, rnn in enumerate(rnn_map['rnns']):
                with tf.variable_scope('rnn{}'.format(rnn_idx)):
                    if 'derived_branches' in rnn_map \
                            and branch_idx < len(rnn_map['derived_branches']) \
                            and rnn_map['derived_branches'][branch_idx]['output_idx'] == rnn_idx:
                        # print("(Rnn._build_rnns)intermediate[rnn_map['derived_branches'][branch_idx]['input_idx']]:",
                        #       intermediate[rnn_map['derived_branches'][branch_idx]['input_idx']])
                        self._build_rnns(
                            intermediate[rnn_map['derived_branches'][branch_idx]['input_idx']],
                            rnn_map['derived_branches'][branch_idx],
                        )
                        branch_idx += 1
                    # print("(Rnn._build_rnns)inp_shape:", inp_shape)
                    rnn.build(inp_shape)
                    inp_shape = tf.TensorShape(inp_shape.as_list()[:-1] + [rnn.num_units])
                    intermediate.append(inp_shape)

    def _add_back_rnn_graph(self, inp, rnn_map, gpu_name, training, saved_state_name, new_state_name):
        if 'derived_branches' in rnn_map:
            rnn_map['derived_branches'] = sorted(rnn_map['derived_branches'], key=lambda x: x['output_idx'])
        state_list = []
        self._prepare_init_state_resback(
            state_list, inp, rnn_map, gpu_name, saved_state_name, new_state_name
        )
        out_dim = rnn_map['rnns'][-1].num_units
        out = tf.zeros(tf.concat([[0], tf.shape(inp)[1:2], [out_dim]], 0), name='loop_output')
        # print("(Rnn._add_back_rnn_graph)inp:", inp)

        if not self._rnns_built:
            self._build_rnns(inp.get_shape(), rnn_map)
            self._rnns_built = True

        def body(inp, out, state):
            # x = inp[0:1, ...]
            with tf.name_scope('prepare_x_and_reduce_input'):
                x = tf.slice(inp, [0, 0, 0], tf.concat([[1], tf.shape(inp)[1:]], 0), name='x')
                # inp = inp[1:, ...]
                inp = tf.slice(
                    inp, [1, 0, 0],
                    tf.concat([tf.shape(inp)[0:1]-1, tf.shape(inp)[1:]], 0), name='loop_input_circumcised'
                )
            x = self._rec_back_rnn_graph(x, state, rnn_map, gpu_name, training, saved_state_name, new_state_name, None)
            out = tf.concat([out, x], 0, name='loop_output_extended')
            return inp, out, state

        def cond(inp, out, state):
            return tf.cast(tf.shape(inp)[0], tf.bool)

        # print(new_state_list)

        _, out, state = tf.while_loop(
            cond,
            body,
            [inp, out, state_list],
            shape_invariants=[
                tf.TensorShape([None, None, inp.get_shape().as_list()[-1]]), tf.TensorShape([None, None, out_dim]),
                tensor_ops.get_shapes(state_list)
            ]
        )
        self._distribute_states(state, rnn_map, new_state_name, gpu_name)
        return out

    def _add_correlation_hooks(self, intermediate):
        # Mean correlation between neurons of hidden state of first LSTM
        self._hooks['correlation'] = tensor_ops.corcov_loss(
            intermediate[0], reduced_axes=[0], cor_axis=2, punish='correlation', reduction='mean',
            norm=self._corcov_norm,
        )
        # Correlation between neurons of hidden state of first LSTM. All values preserved no averaging.
        self._hooks['correlation_values'] = tensor_ops.get_correlation_values(
            intermediate[0],
            reduced_axes=[0],
            cor_axis=2,
        )
        # Correlation between neurons of hidden state of first LSTM
        # and neurons of hidden state of second LSTM. All values preserved no averaging.
        self._hooks['correlation_values_1-2'] = tensor_ops.get_correlation_values_2t(
            intermediate[0],
            intermediate[1],
            reduced_axes=[0],
            cor_axis=2,
        )
        # Mean correlation between neurons of hidden state of second LSTM
        self._hooks['correlation2'] = tensor_ops.corcov_loss(
            intermediate[1], reduced_axes=[0], cor_axis=2, punish='correlation', reduction='mean',
            norm=self._corcov_norm,
        )
        # Mean correlation between neurons of hidden state of first LSTM and
        # neurons of hidden state of second layer
        self._hooks['correlation12'] = tf.reduce_mean(
            tensor_ops.get_correlation_values_2t(
                intermediate[0],
                intermediate[1],
                reduced_axes=[0],
                cor_axis=2,
            ) ** 2
        )
        # self._hooks['correlation_distribution'] = tensor_ops

    def _add_rnn_graph(self, inp, rnn_map, gpu_name, training, saved_state_name, new_state_name):
        if 'derived_branches' in rnn_map:
            rnn_map['derived_branches'] = sorted(rnn_map['derived_branches'], key=lambda x: x['output_idx'])
        branch_idx = 0
        if new_state_name not in rnn_map:
            rnn_map[new_state_name] = dict()
        rnn_map[new_state_name][gpu_name] = list()
        intermediate = list()
        with tf.name_scope(rnn_map['module_name']):
            prepared_state = tensor_ops.prepare_init_state(
                rnn_map[saved_state_name][gpu_name], inp, rnn_map['rnns'], self._network_type)
            # print("(Rnn._add_rnn_graph).prepared_state:", prepared_state)
            for rnn_idx, (rnn, s) in enumerate(zip(rnn_map['rnns'], prepared_state)):
                # print("(Rnn._add_rnn_graph).inp:", inp)
                # print("(Rnn._add_rnn_graph).s:", s)
                if 'derived_branches' in rnn_map \
                        and branch_idx < len(rnn_map['derived_branches']) \
                        and rnn_map['derived_branches'][branch_idx]['output_idx'] == rnn_idx:
                    inp += self._add_rnn_graph(
                        intermediate[rnn_map['derived_branches'][branch_idx]['input_idx']],
                        rnn_map['derived_branches'][branch_idx], gpu_name, training,
                        saved_state_name, new_state_name,
                    )
                    branch_idx += 1
                # with tf.device('/cpu:0'):
                #     ps = []
                #     for t in s:
                #         ps.append(
                #             tf.Print(
                #                 t, list(tensor_stats(t, ['mean', 'variance', 'min', 'max']).values()),
                #                 message="\n\n(Rnn._add_rnn_graph)mean, variance, min, max of "
                #                         "%s in rnn %s:\n" % (t.name, rnn_idx)
                #             )
                #         )
                #
                #     s = tuple(ps)
                inp, new_s = rnn(inp, initial_state=s, training=training)
                if rnn_map['input_idx'] is not None or rnn_idx < len(rnn_map['rnns']) - 1:
                    inp = tf.nn.dropout(inp, keep_prob=1. - self._reg_placeholders['dropout_rate'])
                intermediate.append(inp)
                rnn_map[new_state_name][gpu_name].append(new_s)
            if 'adapter_matrix' in rnn_map:
                inp = tf.einsum('ijk,kl->ijl', inp, rnn_map['adapter_matrix'])
            if 'adapter_bias' in rnn_map:
                inp += rnn_map['adapter_bias']
            # with tf.device('/cpu:0'):
            #     inp = tf.Print(inp, [inp], message="(Rnn._add_rnn_graph)inp (%s):\n" % inp.name)
            # if rnn_map['module_name'] == 'char_enc_dec':
            #     self._hooks['correlation'] = tensor_ops.corcov_loss(
            #         intermediate[0], [0, 1], 2, reduction='mean', norm='abs'
            #     )
            if rnn_map['module_name'] == 'char_enc_dec':
                self._add_correlation_hooks(intermediate)
        return inp

    def _add_rnn_and_output_module(self, embeddings_by_gpu, training):
        logits_by_gpu = list()
        preds_by_gpu = list()
        reset_state_ops = list()
        randomize_state_ops = list()
        if training:
            name_scope = 'train'
        else:
            name_scope = 'inference'
        with tf.name_scope(name_scope):
            for embeddings, gpu_name in zip(embeddings_by_gpu, self._gpu_names):
                with tf.device(gpu_name), tf.name_scope(device_name_scope(gpu_name)):
                    saved_state_name = 'saved_state_' + name_scope
                    new_state_name = 'new_state_' + name_scope
                    self._add_saved_state(self._rnn_map, gpu_name, saved_state_name)
                    with tf.name_scope('rnns'):
                        if self._backward_connections:
                            rnn_res = self._add_back_rnn_graph(
                                embeddings, self._rnn_map, gpu_name, training, saved_state_name, new_state_name
                            )
                        else:
                            rnn_res = self._add_rnn_graph(
                                embeddings, self._rnn_map, gpu_name, training, saved_state_name, new_state_name)
                    # print("(Rnn._add_rnn_and_output_module)rnn_res:", rnn_res)
                    saved_state = self._get_state_from_rnn_map(self._rnn_map, saved_state_name, gpu_name)
                    new_state = self._get_state_from_rnn_map(self._rnn_map, new_state_name, gpu_name)
                    # print("(Rnn._add_rnn_and_output_module)self._rnn_map:", self._rnn_map)
                    reset_state_ops.extend(compose_reset_list(saved_state))
                    randomize_state_ops.extend(compose_randomize_list(
                        saved_state, stddev=self._randomize_state_stddev))
                    save_list = compose_save_list((saved_state, new_state))
                    with tf.control_dependencies(save_list):
                        logits = self._output_module(rnn_res)
                    logits_by_gpu.append(logits)
                    preds_by_gpu.append(tf.nn.softmax(logits))
            reset_state = tf.group(*reset_state_ops)
            randomize_state = tf.group(*randomize_state_ops)
        return logits_by_gpu, preds_by_gpu, reset_state, randomize_state

    def _compute_loss(self, logits, lbls):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.reshape(tf.one_hot(lbls, self._voc_size), [-1, self._voc_size]),
                logits=tf.reshape(logits, [-1, self._voc_size])
            )
        )

    def _compute_loss_and_metrics(self, logits_by_gpu, preds_by_gpu, lbls_by_gpu):
        loss_by_gpu = list()
        additional_metrics = {metric_name: list() for metric_name in self._metrics}
        with tf.name_scope('loss_and_metrics'):
            for logits, preds, lbls, gpu_name in zip(logits_by_gpu, preds_by_gpu, lbls_by_gpu, self._gpu_names):
                with tf.device(gpu_name), tf.name_scope(device_name_scope(gpu_name)):
                    loss = self._compute_loss(logits, lbls)
                    loss_by_gpu.append(loss)
                    add_metrics = compute_metrics_raw_lbls(
                        self._metrics, predictions=preds,
                        labels=lbls, loss=loss, keep_first_dim=False
                    )
                    additional_metrics = append_to_nested(additional_metrics, add_metrics)
            with tf.device(self._base_dev), tf.name_scope('averaging_metrics'):
                average_func = get_average_with_weights_func(self._bs_by_gpu)
                mean_loss = average_func(loss_by_gpu)
                additional_metrics = func_on_list_in_nested(additional_metrics, average_func)
        return mean_loss, loss_by_gpu, additional_metrics

    def _get_train_op(self, loss_by_gpu):
        tower_grads = list()
        for loss, gpu_name in zip(loss_by_gpu, self._gpu_names):
            with tf.device(gpu_name), tf.name_scope(device_name_scope(gpu_name)):
                grads_and_vars = self._optimizer.compute_gradients(loss)
                tower_grads.append(grads_and_vars)
        with tf.device(self._gpu_names[-1]), tf.name_scope('l2_loss_grad'):
            l2_loss = self._l2_loss(tf.get_collection(tf.GraphKeys.WEIGHTS))
            l2_loss_grads_and_vars = self._optimizer.compute_gradients(l2_loss)
        with tf.device(self._base_dev), tf.name_scope(device_name_scope(self._base_dev) + '_gradients'):
            # print("(Rnn._get_train_op)tower_grads:", tower_grads, file=sys.stderr)
            grads_and_vars = average_gradients(tower_grads)
            grads_and_vars_with_l2_loss = list()
            for gv, l2gv in zip(grads_and_vars, l2_loss_grads_and_vars):
                if l2gv[0] is not None:
                    g = gv[0] + l2gv[0]
                else:
                    g = gv[0]
                grads_and_vars_with_l2_loss.append((g, gv[1]))
            grads, v = zip(*grads_and_vars)
            grads, _ = tf.clip_by_global_norm(grads, self._clip_norm)
            train_op = self._optimizer.apply_gradients(zip(grads, v))
        return train_op

    def _build_recursive(
            self, rnn_map, inp_size, out_size,
            preceding_rnn_inp_size, next_rnn_num_units, add_adapter_between_branches=True
    ):
        input_sizes = [inp_size]
        rnns = list()
        back_adapters = self._backward_connections and self._matrix_dim_adjustment
        if back_adapters:
            intermediate_back_adapter_matrices = []
            intermediate_back_adapter_biases = []
        stddevs = tensor_ops.compute_lstm_gru_stddevs(rnn_map['num_nodes'], self._voc_size, self._init_parameter)
        if self._rnn_type == 'lstm':
            Rnn = CudnnLSTM
        elif self._rnn_type == 'gru':
            Rnn = CudnnGRU
        for idx, (nn, stddev) in enumerate(zip(rnn_map['num_nodes'], stddevs)):
            # def init_func(shape, dtype=None):
            #     dtype = tf.float32 if dtype is None else dtype
            #     return tf.truncated_normal(shape, dtype=dtype, stddev=stddev)
            init_truncated_func = lambda shape, dtype: tf.truncated_normal(
                shape, dtype=tf.float32 if dtype is None else dtype, stddev=stddev)
            init_zero_func = lambda shape, dtype: tf.zeros(
                shape, dtype=tf.float32 if dtype is None else dtype)
            rnn = Rnn(
                1,
                nn,
                dropout=self._dropout_rate,
                # kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                kernel_initializer=init_truncated_func,
                bias_initializer=init_zero_func,
                name='%s_%s_%s' % (self._rnn_type, rnn_map['module_name'], idx),
            )
            # rnn.build(inp_shape.concatenate(inp_size))
            rnns.append(rnn)
            if back_adapters and idx > 0:
                intermediate_back_adapter_matrices.append(
                    tf.Variable(
                        tf.zeros([nn, input_sizes[-2]]),
                        name='intermediate_back_adapter_matrix_%s_%s' % (rnn_map['module_name'], idx - 1)
                    )
                )
                intermediate_back_adapter_biases.append(
                    tf.Variable(
                        tf.zeros([input_sizes[-2]]),
                        name='intermediate_back_adapter_bias_%s_%s' % (rnn_map['module_name'], idx - 1)
                    )
                )
            input_sizes.append(nn)
        rnn_map['rnns'] = rnns
        if back_adapters:
            rnn_map['intermediate_back_adapter_matrices'] = intermediate_back_adapter_matrices
            rnn_map['intermediate_back_adapter_biases'] = intermediate_back_adapter_biases
        # print("(Rnn._build_recursive)input_sizes:", input_sizes)
        if add_adapter_between_branches:
            rnn_map['adapter_matrix'] = tf.Variable(
                tf.zeros([rnn_map['num_nodes'][-1], out_size]),
                name='adapter_matrix_%s' % rnn_map['module_name']
            )
            rnn_map['adapter_bias'] = tf.Variable(
                tf.zeros([out_size]),
                name='adapter_bias_%s' % rnn_map['module_name']
            )
            if self._backward_connections:
                rnn_map['out_back_adapter_matrix'] = tf.Variable(
                    tf.zeros([next_rnn_num_units, input_sizes[-2]]),
                    name='out_back_adapter_matrix_%s' % rnn_map['module_name']
                )
                rnn_map['out_back_adapter_bias'] = tf.Variable(
                    tf.zeros([input_sizes[-2]]),
                    name='out_back_adapter_bias_%s' % rnn_map['module_name']
                )

                rnn_map['in_back_adapter_matrix'] = tf.Variable(
                    tf.zeros([input_sizes[1], preceding_rnn_inp_size]),
                    name='in_back_adapter_matrix_%s' % rnn_map['module_name']
                )
                rnn_map['in_back_adapter_bias'] = tf.Variable(
                    tf.zeros([preceding_rnn_inp_size]),
                    name='in_back_adapter_bias_%s' % rnn_map['module_name']
                )
        if 'derived_branches' in rnn_map:
            for branch in rnn_map['derived_branches']:
                self._build_recursive(
                    branch,
                    rnn_map['num_nodes'][branch['input_idx']],
                    rnn_map['num_nodes'][branch['output_idx']-1],
                    input_sizes[branch['input_idx']],
                    input_sizes[branch['output_idx']+1],
                    add_adapter_between_branches=self._matrix_dim_adjustment,
                )

    def _build_rnn_branch_variables(self):
        inp_size = self._emb_size if self._embed_inputs else self._voc_size
        self._build_recursive(self._rnn_map, inp_size, None, None, None, add_adapter_between_branches=False)

    def _build_variables(self):
        with tf.device(self._base_dev), tf.name_scope('build_vars'):
            if self._embed_inputs:
                self._emb_vars = dict(
                    matrix=tf.Variable(
                        tf.truncated_normal(
                            [self._voc_size, self._emb_size],
                            stddev=self._init_parameter * np.sqrt(1. / (self._voc_size + self._emb_size))
                        ),
                        name='embedding_matrix',
                        collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
                    ),
                    bias=tf.Variable(tf.zeros([self._emb_size]), name='embedding_bias')
                )

            self._out_vars = list()
            for layer_idx in range(self._num_out_layers):
                inp_dim, out_dim, stddev = self._compute_output_matrix_parameters(layer_idx)
                self._out_vars.append(
                    dict(
                        matrix=tf.Variable(
                            tf.truncated_normal([inp_dim, out_dim], stddev=stddev),
                            name='output_matrix_%s' % layer_idx,
                            collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
                        ),
                        bias=tf.Variable(
                            tf.zeros([out_dim]),
                            name='output_bias_%s' % layer_idx
                        )
                    )
                )
            self._build_rnn_branch_variables()

    def _get_save_dict_for_rnns(self, rnn_map, accumulated_module_name, module_name=None):
        save_dict = dict()
        if len(accumulated_module_name) > 0:
            accumulated_module_name += '_'
        accumulated_module_name += rnn_map['module_name']
        if module_name is None or rnn_map['module_name'] == module_name:
            for rnn_idx, rnn in enumerate(rnn_map['rnns']):
                save_dict["%s_rnn_%s" % (accumulated_module_name, rnn_idx)] = rnn.saveable
                # print("(Rnn._get_save_dict_for_rnns)rnn.scope_name:", rnn.scope_name)
                # print("(Rnn._get_save_dict_for_rnns)rnn.num_units:", rnn.num_units)
            if 'adapter_matrix' in rnn_map:
                save_dict['%s_adapter_matrix' % accumulated_module_name] = rnn_map['adapter_matrix']
            if 'adapter_bias' in rnn_map:
                save_dict['%s_adapter_bias' % accumulated_module_name] = rnn_map['adapter_bias']
        if 'derived_branches' in rnn_map:
            for branch in rnn_map['derived_branches']:
                save_dict.update(
                    self._get_save_dict_for_rnns(branch, accumulated_module_name, module_name=module_name))
        # for k, v in save_dict.items():
        #     if 'adapter' not in k:
        #         print("(Rnn._get_save_dict_for_rnns)k:", k)
        #         print("(Rnn._get_save_dict_for_rnns)v._OpaqueParamsToCanonical():", v._OpaqueParamsToCanonical())

        return save_dict

    def _get_save_dict_for_base(self):
        save_dict = dict()
        if self._embed_inputs:
            save_dict['embedding_matrix'] = self._emb_vars['matrix']
            save_dict['embedding_bias'] = self._emb_vars['bias']
        for layer_idx, out_core in enumerate(self._out_vars):
            save_dict['output_matrix_%s' % layer_idx] = out_core['matrix']
            save_dict['output_bias_%s' % layer_idx] = out_core['bias']
        return save_dict

    def _create_saver(self):
        # print("(Rnn.create_saver)var_dict:", var_dict)
        save_dict = dict()
        save_dict.update(self._get_save_dict_for_base())
        save_dict.update(self._get_save_dict_for_rnns(self._rnn_map, ""))
        # print("(Rnn._create_saver)save_dict:")
        # for k, v in save_dict.items():
        #     print(k)
        #     print(v, end='\n'*2)
        with tf.device('/cpu:0'):
            saver = tf.train.Saver(save_dict, max_to_keep=None)
        return saver

    def _get_module_names(self, rnn_map):
        names = {rnn_map['module_name']}
        if 'derived_branches' in rnn_map:
            for branch in rnn_map['derived_branches']:
                names |= self._get_module_names(branch)
        return names

    def _create_subgraph_savers(self):
        save_dict = dict()
        save_dict.update(self._get_save_dict_for_base())
        save_dict.update(self._get_save_dict_for_rnns(self._rnn_map, "", module_name=self._rnn_map['module_name']))
        with tf.device('/cpu:0'):
            base_saver = tf.train.Saver(save_dict, max_to_keep=None)
        module_names = self._get_module_names(self._rnn_map)
        module_names.remove(self._rnn_map['module_name'])
        savers = {self._rnn_map['module_name']: base_saver}
        for name in module_names:
            module_save_dict = self._get_save_dict_for_rnns(self._rnn_map, "", module_name=name)
            with tf.device('/cpu:0'):
                savers[name] = tf.train.Saver(module_save_dict, max_to_keep=None)
        return savers

    def _add_inps_and_lbls_phds(self):
        with tf.device(self._base_dev), tf.name_scope('inps_and_lbls'):
            self._inp_and_lbl_phds['inps'] = tf.placeholder(tf.int32, shape=[None, None], name='inps')
            self._inp_and_lbl_phds['lbls'] = tf.placeholder(tf.int32, shape=[None, None], name='lbls')
        self._hooks['inputs'] = self._inp_and_lbl_phds['inps']
        self._hooks['labels'] = self._inp_and_lbl_phds['lbls']
        self._hooks['validation_inputs'] = self._inp_and_lbl_phds['inps']
        self._hooks['validation_labels'] = self._inp_and_lbl_phds['lbls']

    def _add_train_phds(self):
        with tf.device(self._base_dev):
            self._train_phds['learning_rate'] = tf.placeholder(
                tf.float32, name='learning_rate')
            self._hooks['learning_rate'] = self._train_phds['learning_rate']
            if self._optimizer_type in ['momentum', 'nesterov']:
                self._train_phds['momentum'] = tf.placeholder(
                    tf.float32, name='momentum')
                self._hooks['momentum'] = self._train_phds['momentum']

    def _add_reg_placeholders(self):
        with tf.device(self._base_dev):
            self._reg_placeholders['dropout_rate'] = tf.placeholder(tf.float32, name='dropout_rate')
            self._hooks['dropout'] = self._reg_placeholders['dropout_rate']

    def _distribute_by_gpus(self, inputs, labels):
        with tf.device(self._base_dev):
            inputs = tf.split(inputs, self._batch_sizes_on_gpus, 1, name='inp_on_dev')
            inputs_by_device = list()
            for dev_idx, device_inputs in enumerate(inputs):
                inputs_by_device.append(device_inputs)

            labels = tf.split(labels, self._batch_sizes_on_gpus, 1)
            labels_by_device = list()
            for dev_idx, device_labels in enumerate(labels):
                labels_by_device.append(
                    tf.reshape(
                        device_labels,
                        [-1, self._voc_size],
                        name='labels_on_dev_%s' % dev_idx
                    )
                )
            return inputs_by_device, labels_by_device

    def __init__(self, **kwargs):

        if 'rnn_map' in kwargs:
            self._rnn_map = deepcopy(kwargs['rnn_map'])
        elif 'num_nodes' in kwargs:
            self._rnn_map = dict(
                module_name='char_enc_dec',
                num_nodes=kwargs['num_nodes'],
                input_idx=None,
                output_idx=None,
            )
        else:
            self._rnn_map = dict(
                module_name='char_enc_dec',
                num_nodes=[250],
                input_idx=None,
                output_idx=None,
            )
        # print("(Rnn.__init__)self._rnn_map:", self._rnn_map)
        self._rnn_type = kwargs.get('rnn_type', 'lstm')
        self._embed_inputs = kwargs.get('embed_inputs', True)
        self._voc_size = kwargs.get('voc_size', None)
        self._emb_size = kwargs.get('emb_size', 128)
        self._num_out_nodes = kwargs.get('num_out_nodes', [])
        self._num_out_layers = len(self._num_out_nodes) + 1
        self._init_parameter = kwargs.get('init_parameter', 3.)
        self._reg_rate = kwargs.get('reg_rate', 6e-6)
        self._metrics = kwargs.get('metrics', [])
        self._optimizer_type = kwargs.get('optimizer_type', 'adam')
        self._rho = kwargs.get('rho', 0.95)  # used for adadelta
        self._decay = kwargs.get('decay', 0.9)  # used for rmsprop
        self._num_gpus = kwargs.get('num_gpus', 1)
        self._dropout_rate = kwargs.get('dropout_rate', 0.1)  # makes no difference since rnns have one layer
        # print("(Rnn.__init__)self._dropout_rate:", self._dropout_rate)
        self._clip_norm = kwargs.get('clip_norm', 1.)
        self._randomize_state_stddev = kwargs.get('randomize_state_stddev', 0.5)
        self._regime = kwargs.get('regime', 'train')

        self._corcov_norm = kwargs.get('corcov_norm', 'sqr')

        if self._rnn_type == 'lstm':
            self._network_type = 'cudnn_lstm_stacked'
        elif self._rnn_type == 'gru':
            self._network_type = 'cudnn_gru_stacked'
        else:
            self._network_type = None
        self._backward_connections = kwargs.get('backward_connections', False)
        if 'matrix_dim_adjustment' not in kwargs:
            self._matrix_dim_adjustment = not self._backward_connections
        else:
            self._matrix_dim_adjustment = kwargs.get('matrix_dim_adjustment', False)

        # print("(Rnn.__init__)self._network_type:", self._network_type)

        self._hooks = dict(
            inputs=None,
            labels=None,
            train_op=None,
            learning_rate=None,
            momentum=None,
            loss=None,
            predictions=None,
            validation_inputs=None,
            validation_labels=None,
            validation_predictions=None,
            reset_validation_state=None,
            randomize_sample_state=None,
            reset_train_state=None,
            randomize_train_state=None,
            dropout=None,
            saver=None,

            correlation=None,
            correlation_values=None,
            correlation2=None,
            correlation12=None,
        )
        self._hooks['correlation_values_1-2'] = None
        for metric_name in self._metrics:
            self._hooks[metric_name] = None
            self._hooks['validation_' + metric_name] = None

        self._gpu_names = ['/gpu:%s' % i for i in range(self._num_gpus)]
        if self._num_gpus == 1:
            self._base_dev = '/gpu:0'
        else:
            self._base_dev = '/cpu:0'

        if self._embed_inputs:
            self._emb_vars = None
        self._out_vars = None
        self._rnn_branches = dict()
        self._rnns_built = not self._backward_connections

        self._applicable_trainable = dict()

        self._train_storage = dict()
        self._inference_storage = dict()

        self._inp_and_lbl_phds = dict()
        self._train_phds = dict()
        self._inference_placeholders = dict()
        self._reg_placeholders = dict()

        self._add_train_phds()
        self._add_inps_and_lbls_phds()
        self._add_reg_placeholders()
        self._build_variables()

        self._bs_by_gpu = self._get_bs_by_gpu()
        inps_by_gpu = tf.split(self._inp_and_lbl_phds['inps'], self._bs_by_gpu, axis=1)
        lbls_by_gpu = tf.split(self._inp_and_lbl_phds['lbls'], self._bs_by_gpu, axis=1)
        embeddings_by_gpu = self._add_embeding_graph(inps_by_gpu)

        if self._regime == 'train':
            train_logits_by_gpu, train_preds_by_gpu, reset_train_state, randomize_train_state = \
                self._add_rnn_and_output_module(embeddings_by_gpu, True)
            train_loss, train_loss_by_gpu, train_metrics = self._compute_loss_and_metrics(
                train_logits_by_gpu,
                train_preds_by_gpu,
                lbls_by_gpu,
            )
            self._select_optimizer()
            train_op = self._get_train_op(train_loss_by_gpu)

            self._hooks['train_op'] = train_op
            self._hooks['loss'] = train_loss
            self._hooks['predictions'] = tf.concat(train_preds_by_gpu, 1)
            self._hooks['reset_train_state'] = reset_train_state
            self._hooks['randomize_train_state'] = randomize_train_state
            for metric_name in self._metrics:
                self._hooks[metric_name] = train_metrics[metric_name]

        valid_logits_by_gpu, valid_preds_by_gpu, reset_valid_state, randomize_valid_state = \
            self._add_rnn_and_output_module(embeddings_by_gpu, False)

        valid_loss, _, valid_metrics = self._compute_loss_and_metrics(
            valid_logits_by_gpu,
            valid_preds_by_gpu,
            lbls_by_gpu,
        )

        self._hooks['validation_predictions'] = tf.concat(valid_preds_by_gpu, 1)
        self._hooks['reset_validation_state'] = reset_valid_state
        self._hooks['randomize_sample_state'] = randomize_valid_state
        self._hooks['validation_loss'] = valid_loss
        for metric_name in self._metrics:
            self._hooks['validation_' + metric_name] = valid_metrics[metric_name]

        self._hooks['saver'] = self._create_saver()
        self._hooks['subgraph_savers'] = self._create_subgraph_savers()

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)

    def get_building_parameters(self):
        pass
