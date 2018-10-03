import sys
import os
sys.path += [
    os.path.join('/cephfs', os.path.expanduser('~/learning-to-learn')),
    os.path.expanduser('~/learning-to-learn')
]

import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnLSTM as CudnnLSTM
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary, char2vec, pred2vec, \
    pred2vec_fast, vec2char, vec2char_fast, char2id, id2char, get_available_gpus, device_name_scope, \
    average_gradients, InvalidArgumentError, \
    compose_save_list, compose_reset_list, compose_randomize_list, construct_dict_without_none_entries, \
    append_to_nested, get_average_with_weights_func, func_on_list_in_nested
from learning_to_learn.pupils.pupil import Pupil

from learning_to_learn.tensors import compute_metrics_raw_lbls

from helmo.util.tensor import prepare_init_state, get_saved_state_vars, compute_lstm_stddevs


class LstmFastBatchGenerator(object):
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
        return vec2char(vec, vocabulary)

    @staticmethod
    def vec2char_fast(vec, vocabulary):
        return vec2char_fast(vec, vocabulary)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None, random_batch_initiation=False):
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

    def get_num_batches(self):
        return len(self._text) // self._batch_size

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
        # print('(LstmFastBatchGenerator.next)batches[:-1]:', batches[:-1])
        # print('(LstmFastBatchGenerator.next)batches[:-1].shape:', [b.shape for b in batches[:-1]])
        return np.stack(batches[:-1]), np.stack(batches[1:])


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


class Lstm(Pupil):
    _name = 'lstm'

    @classmethod
    def check_kwargs(cls,
                     **kwargs):
        pass

    def get_special_args(self):
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
            input_dim = self._num_nodes[-1]
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
            print('using sgd optimizer')
            self._optimizer = tf.train.GradientDescentOptimizer(
                self._train_phds['learning_rate'])

    def _get_bs_by_gpu(self):
        batch_size = tf.shape(self._inp_and_lbl_phds['inps'])[1]
        bs_on_1st_gpu = batch_size // self._num_gpus
        bs_on_last_gpu = batch_size - (self._num_gpus - 1) * bs_on_1st_gpu
        return [bs_on_1st_gpu] * (self._num_gpus - 1) + [bs_on_last_gpu]

    def _add_embeding_graph(self, inps_by_gpu):
        outputs = list()
        with tf.name_scope('embed'):
            for inp, gpu_name in zip(inps_by_gpu, self._gpu_names):
                with tf.device(gpu_name):
                    with tf.name_scope(device_name_scope(gpu_name)):
                        x = tf.one_hot(inp, self._voc_size, dtype=tf.float32)
                        outputs.append(
                            tf.einsum(
                                'ijk,kl->ijl',
                                x,
                                self._emb_vars['matrix']
                            ) + self._emb_vars['bias']
                        )
        return outputs

    def _add_lstm_and_output_module(self, embeddings_by_gpu, training):
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
                with tf.device(gpu_name):
                    with tf.name_scope(device_name_scope(gpu_name)):
                        saved_state = get_saved_state_vars(len(self._lstms), 'cudnn_stacked')
                        reset_state_ops.extend(compose_reset_list(saved_state))
                        if training:
                            randomize_state_ops.extend(compose_randomize_list(saved_state))
                        state = prepare_init_state(saved_state, embeddings, self._lstms, 'cudnn_stacked')
                        inter = embeddings
                        new_state = list()
                        for lstm, s in zip(self._lstms, state):
                            inter, new_s = lstm(inter, initial_state=s, training=training)
                            new_state.append(new_s)
                        save_list = compose_save_list((saved_state, new_state))
                        with tf.control_dependencies(save_list):
                            logits = self._output_module(inter)
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
                with tf.device(gpu_name):
                    with tf.name_scope(device_name_scope(gpu_name)):
                        loss = self._compute_loss(logits, lbls)
                        loss_by_gpu.append(loss)
                        add_metrics = compute_metrics_raw_lbls(
                            self._metrics, predictions=preds,
                            labels=lbls, loss=loss, keep_first_dim=False
                        )
                        additional_metrics = append_to_nested(additional_metrics, add_metrics)
            with tf.device(self._base_dev):
                with tf.name_scope('averaging_metrics'):
                    with tf.name_scope(device_name_scope(self._base_dev)):
                        average_func = get_average_with_weights_func(self._bs_by_gpu)
                        mean_loss = average_func(loss_by_gpu)
                        additional_metrics = func_on_list_in_nested(additional_metrics, average_func)
        return mean_loss, loss_by_gpu, additional_metrics

    def _get_train_op(self, loss_by_gpu):
        tower_grads = list()
        for loss, gpu_name in zip(loss_by_gpu, self._gpu_names):
            with tf.device(gpu_name):
                with tf.name_scope(device_name_scope(gpu_name)):
                    grads_and_vars = self._optimizer.compute_gradients(loss)
                    tower_grads.append(grads_and_vars)
        with tf.device(self._gpu_names[-1]):
            with tf.name_scope('l2_loss_grad'):
                l2_loss = self._l2_loss(tf.get_collection(tf.GraphKeys.WEIGHTS))
                l2_loss_grads_and_vars = self._optimizer.compute_gradients(l2_loss)
        with tf.device(self._base_dev):
            with tf.name_scope(device_name_scope(self._base_dev) + '_gradients'):
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

    # def _train_graph(self):
    #     inputs, labels = self._prepare_inputs_and_labels(
    #         self._train_inputs_and_labels_placeholders['inputs'], self._train_inputs_and_labels_placeholders['labels'])
    #     inputs_by_device, labels_by_device = self._distribute_by_gpus(inputs, labels)
    #     trainable = self._applicable_trainable
    #     tower_grads = list()
    #     preds = list()
    #     losses = list()
    #     reset_state_ops = list()
    #
    #     additional_metrics = dict()
    #     for add_metric in self._additional_metrics:
    #         additional_metrics[add_metric] = list()
    #
    #     with tf.name_scope('train'):
    #         for gpu_batch_size, gpu_name, device_inputs, device_labels in zip(
    #                 self._batch_sizes_on_gpus, self._gpu_names, inputs_by_device, labels_by_device):
    #             with tf.device(gpu_name):
    #                 with tf.name_scope(device_name_scope(gpu_name)):
    #                     saved_states = list()
    #                     for layer_idx, layer_num_nodes in enumerate(self._num_nodes):
    #                         saved_states.append(
    #                             (tf.Variable(
    #                                 tf.zeros([gpu_batch_size, layer_num_nodes]),
    #                                 trainable=False,
    #                                 name='saved_state_%s_%s' % (layer_idx, 0)),
    #                              tf.Variable(
    #                                  tf.zeros([gpu_batch_size, layer_num_nodes]),
    #                                  trainable=False,
    #                                  name='saved_state_%s_%s' % (layer_idx, 1)))
    #                         )
    #                     reset_state_ops.extend(compose_reset_list(saved_states))
    #
    #                     all_states = saved_states
    #                     embeddings, _ = self._embed(device_inputs, trainable['embedding_matrix'])
    #                     rnn_outputs, all_states, _ = self._rnn_module(
    #                         embeddings, all_states, trainable['lstm_matrices'],
    #                         trainable['lstm_biases'])
    #                     logits, _ = self._output_module(
    #                         rnn_outputs, trainable['output_matrices'], trainable['output_biases'])
    #
    #                     save_ops = compose_save_list((saved_states, all_states))
    #
    #                     with tf.control_dependencies(save_ops):
    #                         all_matrices = [trainable['embedding_matrix']]
    #                         all_matrices.extend(trainable['lstm_matrices'])
    #                         all_matrices.extend(trainable['output_matrices'])
    #                         l2_loss = self._l2_loss(all_matrices)
    #
    #                         loss = tf.reduce_mean(
    #                             tf.nn.softmax_cross_entropy_with_logits(labels=device_labels, logits=logits))
    #                         concat_pred = tf.nn.softmax(logits)
    #
    #                         add_metrics = compute_metrics(
    #                             self._additional_metrics, predictions=concat_pred,
    #                             labels=device_labels, loss=loss, keep_first_dim=False
    #                         )
    #
    #                         preds.append(tf.split(concat_pred, self._num_unrollings))
    #
    #                         losses.append(loss)
    #
    #                         grads_and_vars = opt.compute_gradients(loss + l2_loss)
    #                         tower_grads.append(grads_and_vars)
    #                         additional_metrics = append_to_nested(additional_metrics, add_metrics)
    #
    #                         # splitting concatenated results for different characters
    #         with tf.device(self._base_dev):
    #             with tf.name_scope(device_name_scope(self._base_dev) + '_gradients'):
    #                 # print('(Lstm._train_graph)tower_grads:', tower_grads)
    #                 grads_and_vars = average_gradients(tower_grads)
    #                 # with tf.device('/cpu:0'):
    #                 #     grads_and_vars = [
    #                 #         (
    #                 #             tf.Print(
    #                 #                 grad, [tf.nn.l2_loss(grad), tf.nn.l2_loss(var)],
    #                 #                 message="l2_loss(grad(%s)) and l2_loss(%s):\n" % (var.name, var.name)
    #                 #             ),
    #                 #             var
    #                 #         )
    #                 #         for grad, var in grads_and_vars
    #                 #     ]
    #                 grads, v = zip(*grads_and_vars)
    #                 # grads, _ = tf.clip_by_global_norm(grads, 1.)
    #                 self.train_op = opt.apply_gradients(zip(grads, v))
    #                 self._hooks['train_op'] = self.train_op
    #                 self._hooks['reset_pupil_train_state'] = tf.group(*reset_state_ops)
    #                 # composing predictions
    #                 preds_by_char = list()
    #                 # print('preds:', preds)
    #                 for one_char_preds in zip(*preds):
    #                     # print('one_char_preds:', one_char_preds)
    #                     preds_by_char.append(tf.concat(one_char_preds, 0))
    #                 # print('len(preds_by_char):', len(preds_by_char))
    #                 self.predictions = tf.concat(preds_by_char, 0)
    #                 self._hooks['predictions'] = self.predictions
    #                 # print('self.predictions.get_shape().as_list():', self.predictions.get_shape().as_list())
    #                 average_func = get_average_with_weights_func(self._batch_sizes_on_gpus)
    #                 self.loss = average_func(losses)
    #                 additional_metrics = func_on_list_in_nested(additional_metrics, average_func)
    #                 self._hooks['loss'] = self.loss
    #                 for k, v in additional_metrics.items():
    #                     self._hooks[k] = v
    #
    # def _validation_graph(self):
    #     trainable = self._applicable_trainable
    #     with tf.device(self._gpu_names[0]):
    #         with tf.name_scope('validation'):
    #             self.validation_labels = tf.placeholder(tf.int32, [1, 1])
    #             self.sample_input = tf.placeholder(tf.int32,
    #                                                shape=[1, 1, 1],
    #                                                name='sample_input')
    #             inputs = tf.reshape(self.sample_input, [1, -1])
    #             sample_input = tf.one_hot(inputs, self._vocabulary_size)
    #             labels = tf.reshape(self.validation_labels, [1])
    #             validation_labels_prepared = tf.one_hot(labels, self._vocabulary_size)
    #
    #             self._hooks['validation_inputs'] = self.sample_input
    #             self._hooks['validation_labels'] = self.validation_labels
    #             self._hooks['validation_labels_prepared'] = validation_labels_prepared
    #             saved_sample_state = list()
    #             for layer_idx, layer_num_nodes in enumerate(self._num_nodes):
    #                 saved_sample_state.append(
    #                     (tf.Variable(
    #                         tf.zeros([1, layer_num_nodes]),
    #                         trainable=False,
    #                         name='saved_sample_state_%s_%s' % (layer_idx, 0)),
    #                      tf.Variable(
    #                          tf.zeros([1, layer_num_nodes]),
    #                          trainable=False,
    #                          name='saved_sample_state_%s_%s' % (layer_idx, 1)))
    #                 )
    #
    #             reset_list = compose_reset_list(saved_sample_state)
    #
    #             self.reset_sample_state = tf.group(*reset_list)
    #             self._hooks['reset_validation_state'] = self.reset_sample_state
    #
    #             randomize_list = compose_randomize_list(saved_sample_state)
    #
    #             self.randomize = tf.group(*randomize_list)
    #             self._hooks['randomize_sample_state'] = self.randomize
    #
    #             embeddings, _ = self._embed(sample_input, trainable['embedding_matrix'])
    #             # print('embeddings:', embeddings)
    #             rnn_output, sample_state, _ = self._rnn_module(
    #                 embeddings, saved_sample_state, trainable['lstm_matrices'],
    #                 trainable['lstm_biases'])
    #             sample_logit, _ = self._output_module(
    #                 rnn_output, trainable['output_matrices'], trainable['output_biases'])
    #
    #             sample_save_ops = compose_save_list((saved_sample_state, sample_state))
    #
    #             with tf.control_dependencies(sample_save_ops):
    #                 self.sample_prediction = tf.nn.softmax(sample_logit)
    #                 metrics = compute_metrics(
    #                     self._additional_metrics + ['loss'],
    #                     predictions=self.sample_prediction,
    #                     labels=validation_labels_prepared,
    #                     keep_first_dim=False
    #                 )
    #
    #                 self._hooks['validation_predictions'] = self.sample_prediction
    #                 for k, v in metrics.items():
    #                     self._hooks['validation_' + k] = v

    def _declare_lstms(self):
        if isinstance(self._num_nodes, int) or self._num_nodes == [self._num_nodes[0]] * self._num_layers:
            if isinstance(self._num_nodes, int):
                nn = self._num_nodes
            else:
                nn = self._num_nodes[0]
            stddevs = compute_lstm_stddevs([nn], self._voc_size, self._init_parameter)
            self._lstms = [
                CudnnLSTM(
                    self._num_layers,
                    nn,
                    dropout=self._dropout_rate,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs[0]),
                    name='lstm_0',
                )
            ]
        else:
            if len(self._num_nodes) != self._num_layers:
                raise InvalidArgumentError(
                    "If 'num_nodes' is list its, length have to be equal to 'num_layers'",
                    self._num_nodes,
                    'self._num_nodes',
                    "len(self._num_nodes) == self._num_layers"
                )
            stddevs = compute_lstm_stddevs(self._num_nodes, self._voc_size, self._init_parameter)
            self._lstms = list()
            for nn, stddev in zip([self._voc_size] + self._num_nodes, stddevs):
                self._lstms.append(
                    CudnnLSTM(
                        1,
                        nn,
                        dropout=self._dropout_rate,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                        name='lstm_0',
                    )
                )

    def _build_variables(self):
        with tf.device(self._base_dev):
            with tf.name_scope('build_vars'):

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
                                tf.truncated_normal([inp_dim, out_dim],
                                stddev=stddev),
                                name='output_matrix_%s' % layer_idx,
                                collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
                            ),
                            bias=tf.Variable(
                                tf.zeros([out_dim]),
                                name='output_bias_%s' % layer_idx
                            )
                        )
                    )

                inp_shape = self._inp_and_lbl_phds['inps'].shape
                inp_size = self._emb_size
                for lstm, nn in zip(self._lstms, self._num_nodes):
                    lstm.build(inp_shape.concatenate([inp_size]))
                    inp_size = nn

    def create_saver(self):
        # print("(Lstm.create_saver)var_dict:", var_dict)
        with tf.device('/cpu:0'):
            saved_vars = dict()
            saved_vars['embedding_matrix'] = self._emb_vars['matrix']
            saved_vars['embedding_matrix'] = self._emb_vars['bias']
            for lstm_idx, lstm in enumerate(self._lstms):
                saved_vars['lstm_%s' % lstm_idx] = lstm.saveable
            for layer_idx, out_core in enumerate(self._out_vars):
                saved_vars['output_matrix_%s' % layer_idx] = out_core['matrix']
                saved_vars['output_bias_%s' % layer_idx] = out_core['bias']
            saver = tf.train.Saver(saved_vars, max_to_keep=None)
        return saver

    def _add_inps_and_lbls_phds(self):
        with tf.device(self._base_dev):
            with tf.name_scope('inps_and_lbls'):
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

        self._num_layers = kwargs.get('num_layers', 2)
        self._num_nodes = kwargs.get('num_nodes', [112, 113])
        self._voc_size = kwargs.get('voc_size', None)
        self._emb_size = kwargs.get('emb_size', 128)
        self._num_out_layers = kwargs.get('num_out_layers', 1)
        self._num_out_nodes = kwargs.get('num_out_nodes', [])
        self._init_parameter = kwargs.get('init_parameter', 3.)
        self._reg_rate = kwargs.get('reg_rate', 6e-6)
        self._metrics = kwargs.get('metrics', [])
        self._optimizer_type = kwargs.get('optimizer_type', 'adam')
        self._rho = kwargs.get('rho', 0.95)  # used for adadelta
        self._decay = kwargs.get('decay', 0.9)  # used for rmsprop
        self._num_gpus = kwargs.get('num_gpus', 1)
        self._dropout_rate = kwargs.get('dropout_rate', 0.9)
        self._clip_norm = kwargs.get('clip_norm', 1.)
        self._regime = kwargs.get('regime', 'train')

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
            dropout=None,
            saver=None)
        for metric_name in self._metrics:
            self._hooks[metric_name] = None
            self._hooks['validation_' + metric_name] = None

        self._gpu_names = ['/gpu:%s' % i for i in range(self._num_gpus)]
        if self._num_gpus == 1:
            self._base_dev = '/gpu:0'
        else:
            self._base_dev = '/cpu:0'

        self._emb_vars = None
        self._out_vars = None
        self._lstms = None

        self._applicable_trainable = dict()

        self._train_storage = dict()
        self._inference_storage = dict()

        self._inp_and_lbl_phds = dict()
        self._train_phds = dict()
        self._inference_placeholders = dict()

        self._add_train_phds()
        self._add_inps_and_lbls_phds()
        self._declare_lstms()
        self._build_variables()

        self._bs_by_gpu = self._get_bs_by_gpu()
        inps_by_gpu = tf.split(self._inp_and_lbl_phds['inps'], self._bs_by_gpu, axis=1)
        lbls_by_gpu = tf.split(self._inp_and_lbl_phds['lbls'], self._bs_by_gpu, axis=1)
        embeddings_by_gpu = self._add_embeding_graph(inps_by_gpu)
        train_logits_by_gpu, train_preds_by_gpu, reset_train_state, _ = \
            self._add_lstm_and_output_module(embeddings_by_gpu, True)
        valid_logits_by_gpu, valid_preds_by_gpu, reset_valid_state, randomize_valid_state = \
            self._add_lstm_and_output_module(embeddings_by_gpu, False)
        train_loss, train_loss_by_gpu, train_metrics = self._compute_loss_and_metrics(
            train_logits_by_gpu,
            train_preds_by_gpu,
            lbls_by_gpu,
        )
        valid_loss, _, valid_metrics = self._compute_loss_and_metrics(
            valid_logits_by_gpu,
            valid_preds_by_gpu,
            lbls_by_gpu,
        )
        if self._regime == 'train':
            self._select_optimizer()
            train_op = self._get_train_op(train_loss_by_gpu)
            self._hooks['train_op'] = train_op

        self._hooks['loss'] = train_loss
        self._hooks['predictions'] = tf.concat(train_preds_by_gpu, 1)
        self._hooks['validation_predictions'] = tf.concat(valid_preds_by_gpu, 1)
        self._hooks['reset_validation_state'] = reset_valid_state
        self._hooks['randomize_sample_state'] = randomize_valid_state
        self._hooks['validation_loss'] = valid_loss
        for metric_name in self._metrics:
            self._hooks[metric_name] = train_metrics[metric_name]
            self._hooks['validation_' + metric_name] = valid_metrics[metric_name]

        self._hooks['saver'] = self.create_saver()

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)

    def get_building_parameters(self):
        pass
