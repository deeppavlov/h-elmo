import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnLSTM as CudnnLSTM
from tensorflow.nn.rnn_cell import LSTMCell as LSTMCell
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel


def add_cudnn_lstm(inps, state, num_layers, num_units):
    lstm = CudnnLSTM(num_layers, num_units, input_mode='linear_input', )
    output, state = lstm(inps, initial_state=state)
    return output, state


def add_stacked_cudnn_lstm(inps, state, num_layers, num_units):
    lstms = [CudnnLSTM(1, num_units, input_mode='linear_input', ) for _ in range(num_layers)]
    inter = inps
    new_state = list()
    for lstm, s in zip(lstms, state):
        inter, new_s = lstm(inter, initial_state=s)
        new_state.append(s)
    return inter, new_state


def add_cell_lstm(inps, state, num_layers, num_units):
    lstms = [LSTMCell(num_units, dtype=tf.float32) for _ in range(num_layers)]
    multilayer_lstm = tf.contrib.rnn.MultiRNNCell(lstms)
    zero_state = multilayer_lstm.zero_state(tf.shape(inps)[1], tf.float32)
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
        # max length of output sequence
        self.max_length = kwargs.get('max_length', 20)
        self.grad_clip = kwargs.get('grad_clip', 5.0)
        self.vocab_size = kwargs.get('vocab_size', 100)
        self.init_parameter = kwargs.get('init_parameter', 1.)
        self.device = kwargs.get('device_type', 'gpu')

        # create tensorflow session to run computational graph in it
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self.add_lstm_to_graph()

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

    def add_lstm_graph(self):
        self.input_projection_matrix = tf.Variable(
            tf.truncated_normal(
                [self.vocab_size, self.projection_size],
                stddev=self.init_parameter / (self.vocab_size + self.projection_size)**0.5
            )
        )
        self.output_projection_matrix = tf.Variable(
            tf.truncated_normal(
                [self.num_units[-1], self.vocab_size],
                stddev=self.init_parameter / (self.num_units[-1] + self.vocab_size)**0.5
            )
        )
        self.add_placeholders()

        x_proj = tf.matmul(self.x_ph, self.projection_matrix)
        if self.device == 'gpu':
            if all([nu == self.num_units[i+1] for i, nu in enumerate(self.num_units[:-1])]):
                lstm_output, state = add_cudnn_lstm(x_proj, self.num_layers, self.num_units[0])
            else:
                lstm_output, state = add_stacked_cudnn_lstm(x_proj, self.num_layers, self.num_units)
        else:
            lstm_output = add_cell_lstm(x_proj, self.num_layers, self.num_units)
        logits =




