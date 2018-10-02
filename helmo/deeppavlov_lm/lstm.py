import sys
sys.path.append('/home/anton/dpenv/src/deeppavlov')

import tensorflow as tf

from tensorflow.contrib.cudnn_rnn import CudnnLSTM as CudnnLSTM
from tensorflow.nn.rnn_cell import LSTMCell as LSTMCell
from tensorflow.nn.rnn_cell import LSTMStateTuple as LSTMStateTuple
# from deeppavlov.core.common.registry import register
# from deeppavlov.core.models.tf_model import TFModel

import sys
from helmo.util.deal_with_cephfs import add_repo_2_sys_path
sys.path = add_repo_2_sys_path('DeepPavlov')


from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger

from helmo.util.tensor import prepare_init_state, compose_save_list, compute_lstm_stddevs, get_saved_state_vars

logger = get_logger(__name__)


def add_cudnn_lstm(inps, state, num_layers, num_units, input_dim, init_parameter):
    input_dim = max(input_dim, num_units)
    stddevs = compute_lstm_stddevs([num_units], input_dim, init_parameter)
    # print("(add_cudnn_lstm)stddevs:", stddevs)
    lstm = CudnnLSTM(
        num_layers, num_units, input_mode='linear_input',
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs[0])
    )
    state = prepare_init_state(state, inps, lstm, 'cudnn')
    output, state = lstm(inps, initial_state=state)
    return output, state


def add_stacked_cudnn_lstm(inps, state, num_units, input_dim, init_parameter):
    stddevs = compute_lstm_stddevs(num_units, input_dim, init_parameter)
    lstms = [
        CudnnLSTM(1, nu, input_mode='linear_input', kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        for nu, stddev in zip(num_units, stddevs)
    ]
    state = prepare_init_state(state, inps, lstms, 'cudnn_stacked')
    inter = inps
    new_state = list()
    # print("(add_stacked_cudnn_lstm)state:", state)
    for lstm, s in zip(lstms, state):
        inter, new_s = lstm(inter, initial_state=s)
        new_state.append(new_s)
    # print("(add_stacked_cudnn_lstm)new_state:", new_state)
    return inter, new_state


def add_cell_lstm(inps, state, num_units, input_dim, init_parameter):
    stddevs = compute_lstm_stddevs(num_units, input_dim, init_parameter)
    lstms = [
        LSTMCell(
            nu, dtype=tf.float32, state_is_tuple=False,
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        for nu, stddev in zip(num_units, stddevs)
    ]
    multilayer_lstm = tf.contrib.rnn.MultiRNNCell(lstms, state_is_tuple=False)
    # print("(add_cell_lstm)state:", state)
    # print("(add_cell_lstm)multilayer_lstm.state_size:", multilayer_lstm.state_size)
    state = prepare_init_state(state, inps, multilayer_lstm, 'cell')
    if state is None:
        state = multilayer_lstm.zero_state(tf.shape(inps)[1], tf.float32)

    # print("(add_cell_lstm)multilayer_lstm.state:", multilayer_lstm.state)
    output, state = tf.nn.dynamic_rnn(
        multilayer_lstm, inps, initial_state=state, parallel_iterations=1024, time_major=True
    )
    return output, state


@register('lstm')
class LSTM(TFModel):
    def __init__(self, **kwargs):
        # hyperparameters

        # dimension of word embeddings
        self.projection_size = kwargs.get('projection_size', 100)
        self.num_layers = kwargs.get('num_layers', 1)
        # size of recurrent cell in encoder and decoder
        self.num_units = kwargs.get('num_units', 100)
        if isinstance(self.num_units, int):
            self.num_units = [self.num_units] * self.num_layers
        # dropout keep_probability
        self.keep_prob = kwargs.get('keep_prob', 0.8)
        # learning rate
        self.learning_rate = kwargs.get('learning_rate', 3e-04)
        self.learning_rate_patience = kwargs.get('learning_rate_patience', -1)
        self.lr_decay_factor = kwargs.get('lr_decay_factor', .5)
        # max length of output sequence
        self.max_length = kwargs.get('max_length', 20)
        self.grad_clip = kwargs.get('grad_clip', 5.0)
        self.vocab_size = kwargs.get('vocab_size', 100)
        if self.vocab_size == 80:
            raise KeyboardInterrupt
        self.init_parameter = kwargs.get('init_parameter', 1.)
        self.optimizer_type = kwargs.get('optimizer', 'adam')
        self.device = kwargs.get('device_type', 'gpu')

        self.last_impatience = 0
        self.lr_impatience = 0

        # create tensorflow session to run computational graph in it
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self.add_lstm_graph()

        # define train op
        if self.optimizer_type == 'adam':
            opt = tf.train.AdamOptimizer
        elif self.optimizer_type == 'sgd':
            opt = tf.train.GradientDescentOptimizer
        self.train_op = self.get_train_op(self.loss, self.lr_ph,
                                          optimizer=opt,
                                          clip_norm=self.grad_clip)
        # initialize graph variables
        self.sess.run(tf.global_variables_initializer())

        super().__init__(**kwargs)
        # load saved model if there is one
        if self.load_path is not None:
            self.load()

    def add_lstm_graph(self):
        self.add_trainable_vars()
        self.add_placeholders()

        x = tf.one_hot(tf.transpose(self.x_ph), self.vocab_size)
        y = tf.one_hot(tf.transpose(self.y_ph), self.vocab_size)

        x_embedded = tf.tensordot(x, self.input_layer_matrix, axes=[[-1], [0]]) + self.input_layer_bias
        if self.device == 'gpu':
            if all([nu == self.num_units[i+1] for i, nu in enumerate(self.num_units[:-1])]):
                lstm_type = 'cudnn'
            else:
                lstm_type = 'cudnn_stacked'
        else:
            lstm_type = 'cell'
        # lstm_type = 'cell'
        saved_state = get_saved_state_vars(self.num_layers, lstm_type)
        # saved_state = list(saved_state)
        # for idx, s in enumerate(saved_state):
        #     x_embedded = tf.Print(x_embedded, [tf.shape(s)], message="(add_lstm_graph)saved_state[%s].shape:\n" % idx)
        # saved_state = tuple(saved_state)
        if lstm_type == 'cudnn':
            lstm_output, state = add_cudnn_lstm(
                x_embedded, saved_state, self.num_layers, self.num_units[0], self.projection_size, self.init_parameter)
        elif lstm_type == 'cudnn_stacked':
            lstm_output, state = add_stacked_cudnn_lstm(
                x_embedded, saved_state, self.num_units, self.projection_size, self.init_parameter)
        elif lstm_type == 'cell':
            lstm_output, state = add_cell_lstm(
                x_embedded, saved_state, self.num_units, self.projection_size, self.init_parameter)

        logits = tf.tensordot(lstm_output, self.softmax_layer_matrix, axes=[[-1], [0]]) + self.softmax_layer_bias

        save_list = compose_save_list((saved_state, state))
        # print("(LSTM.add_lstm_graph)save_list:", save_list)
        with tf.control_dependencies(save_list):
            self.predictions = tf.transpose(tf.argmax(tf.nn.softmax(logits), axis=-1))
            # print(self.predictions.get_shape().as_list())
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=y,
                    logits=logits,
                )
            )
            # opt = tf.train.AdamOptimizer(1)
            # grads_and_vars = opt.compute_gradients(self.loss)
            # print(grads_and_vars)

    def add_trainable_vars(self):
        self.input_layer_matrix = tf.Variable(
            tf.truncated_normal(
                [self.vocab_size, self.projection_size],
                stddev=self.init_parameter / (self.vocab_size + self.projection_size) ** 0.5
            ),
            name='input_layer_matrix',
        )
        self.input_layer_bias = tf.Variable(
            tf.zeros(
                [self.projection_size]
            ),
            name='input_layer_bias',
        )
        self.softmax_layer_matrix = tf.Variable(
            tf.truncated_normal(
                [self.num_units[-1], self.vocab_size],
                stddev=self.init_parameter / (self.num_units[-1] + self.vocab_size) ** 0.5
            ),
            name='softmax_layer_matrix',
        )
        self.softmax_layer_bias = tf.Variable(
            tf.zeros(
                [self.vocab_size]
            ),
            name='softmax_layer_bias',
        )

    def add_placeholders(self):
        # placeholders for inputs
        self.x_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='x_ph')
        self.y_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_ph')

        # placeholders for model parameters
        self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name='lr_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')

    def _build_feed_dict(self, x, y=None):
        feed_dict = {
            self.x_ph: x,
        }
        if y is not None:
            feed_dict.update({
                self.y_ph: y,
                self.lr_ph: self.learning_rate,
                self.keep_prob_ph: self.keep_prob,
            })
        return feed_dict

    def train_on_batch(self, x, y):
        feed_dict = self._build_feed_dict(x, y)
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        # print("(LSTM.train_on_batch)loss:", loss)
        return loss

    def __call__(self, x):
        # print("(LSTM.__call__)x:", x)
        feed_dict = self._build_feed_dict(x)
        y_pred = self.sess.run(self.predictions, feed_dict=feed_dict)
        # print("(LSTM.__call__)y_pred:", y_pred)
        return y_pred

    def process_event(self, event_name: str, data) -> None:
        """
        Processes events sent by trainer. Implements learning rate decay.

        Args:
            event_name: event_name sent by trainer
            data: number of examples, epochs, metrics sent by trainer
        """
        if event_name == "after_validation" and self.last_impatience >= 0:
            if data['impatience'] > self.last_impatience:
                self.lr_impatience += 1
            else:
                self.lr_impatience = 0

            self.last_impatience = data['impatience']

            if self.lr_impatience >= self.learning_rate_patience:
                self.lr_impatience = 0
                self.learning_rate *= self.lr_decay_factor
                logger.info('LSTM model: learning_rate changed to {}'.format(self.learning_rate))
            logger.info(
                'LSTM model: lr_impatience: {}, learning_rate: {}'.format(self.lr_impatience, self.learning_rate)
            )
