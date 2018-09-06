import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnLSTM as CudnnLSTM
from tensorflow.nn.rnn_cell import LSTMCell as LSTMCell
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel


def synchronous_flatten(*nested):
    if not isinstance(nested[0], (tuple, list, dict)):
        return [[n] for n in nested]
    output = [list() for _ in nested]
    if isinstance(nested[0], dict):
        for k in nested[0].keys():
            flattened = synchronous_flatten(*[n[k] for n in nested])
            for o, f in zip(output, flattened):
                o.extend(f)
    else:
        try:
            for inner_nested in zip(*nested):
                flattened = synchronous_flatten(*inner_nested)
                for o, f in zip(output, flattened):
                    o.extend(f)
        except TypeError:
            print('(synchronous_flatten)nested:', nested)
            raise
    return output


def compose_save_list(*pairs, name_scope='save_list'):
    with tf.name_scope(name_scope):
        save_list = list()
        for pair in pairs:
            # print('pair:', pair)
            [variables, new_values] = synchronous_flatten(pair[0], pair[1])
            # print("(useful_functions.compose_save_list)variables:", variables)
            # variables = flatten(pair[0])
            # # print(variables)
            # new_values = flatten(pair[1])
            for variable, value in zip(variables, new_values):
                save_list.append(tf.assign(variable, value, validate_shape=False))
        return save_list


def deep_zip(objects, depth, permeable_types=(list, tuple, dict)):
    if depth != 0 and isinstance(objects[0], permeable_types):
        if isinstance(objects[0], (list, tuple)):
            zipped = list()
            for comb in zip(*objects):
                zipped.append(
                    deep_zip(comb, depth-1, permeable_types=permeable_types)
                )
            if isinstance(objects[0], tuple):
                zipped = tuple(zipped)
            return zipped
        elif isinstance(objects[0], dict):
            zipped = dict()
            for key in objects[0].keys():
                values = [obj[key] for obj in objects]
                zipped[key] = deep_zip(values, depth-1, permeable_types=permeable_types)
            return zipped
    return tuple(objects)


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


def is_scalar_tensor(t):
    return tf.equal(tf.shape(tf.shape(t))[0], 0)


def replace_empty_saved_state(saved_and_zero_state):
    return tf.cond(
        is_scalar_tensor(saved_and_zero_state[0]),
        true_fn=lambda: saved_and_zero_state[1],
        false_fn=lambda: saved_and_zero_state[0],
    )


def prepare_init_state(saved_state, inps, lstms, lstm_type):
    if saved_state is None:
        return None
    zero_state = get_zero_state(inps, lstms, lstm_type)
    zero_and_saved_states_zipped = deep_zip(
        [saved_state, zero_state], -1
    )
    if lstm_type == 'cudnn_stacked':
        depth = 2
    else:
        depth = 1
    return apply_func_on_depth(zero_and_saved_states_zipped, replace_empty_saved_state, depth)


def add_cudnn_lstm(inps, state, num_layers, num_units):
    lstm = CudnnLSTM(num_layers, num_units, input_mode='linear_input', )
    state = prepare_init_state(state, inps, lstm, 'cudnn')
    output, state = lstm(inps, initial_state=state)
    return output, state


def add_stacked_cudnn_lstm(inps, state, num_layers, num_units):
    lstms = [CudnnLSTM(1, num_units, input_mode='linear_input', ) for _ in range(num_layers)]
    state = prepare_init_state(state, inps, lstms, 'cudnn_stacked')
    inter = inps
    new_state = list()
    for lstm, s in zip(lstms, state):
        inter, new_s = lstm(inter, initial_state=s)
        new_state.append(s)
    return inter, new_state


def add_cell_lstm(inps, state, num_layers, num_units):
    lstms = [LSTMCell(num_units, dtype=tf.float32) for _ in range(num_layers)]
    multilayer_lstm = tf.contrib.rnn.MultiRNNCell(lstms)
    state = prepare_init_state(state, inps, multilayer_lstm, 'cell')
    if state is None:
        state = multilayer_lstm.zero_state(tf.shape(inps)[1], tf.float32)
    output, state = tf.nn.dynamic_rnn(
        multilayer_lstm, inps, initial_state=state, parallel_iterations=1024, time_major=True
    )
    return output, state


def cudnn_cell_zero_state(batch_size, cell):
    zero_state = tuple(
        [
            tf.zeros(
                tf.concat(
                    [
                        [cell.num_dirs * cell.num_layers],
                        tf.reshape(batch_size, [1]),
                        [cell.num_units]
                    ],
                    0
                )
            )
        ] * 2
    )
    return zero_state


def get_zero_state(inps, lstms, lstm_type):
    batch_size = tf.shape(inps)[1]
    if lstm_type == 'cell':
        zero_state = lstms.zero_state(batch_size, tf.float32)
    if lstm_type == 'cudnn':
        zero_state = cudnn_cell_zero_state(batch_size, lstms)
    if lstm_type == 'cudnn_stacked':
        zero_state = [cudnn_cell_zero_state(batch_size, lstm) for lstm in lstms]
    return zero_state


def get_saved_state_vars(num_layers, lstm_type):
    if lstm_type == 'cudnn':
        state = (
            tf.Variable(0., trainable=False, validate_shape=False),
            tf.Variable(0., trainable=False, validate_shape=False),
        )
    elif lstm_type == 'cudnn_stacked':
        state = [
            (
                tf.Variable(0., trainable=False, validate_shape=False),
                tf.Variable(0., trainable=False, validate_shape=False),
            ) for _ in range(num_layers)
        ]
    elif lstm_type == 'cell':
        state = tuple(
            [tf.Variable(0., trainable=False, validate_shape=False) for _ in range(num_layers*2)]
        )
    else:
        state = None
    return state


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

        self.add_lstm_graph()

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
        self.add_trainable_vars()
        self.add_placeholders()

        x = tf.one_hot(self.x_ph, self.vocab_size)
        y = tf.one_hot(self.y_ph, self.vocab_size)

        x_embedded = tf.matmul(x, self.projection_matrix) + self.input_layer_bias
        if self.device == 'gpu':
            if all([nu == self.num_units[i+1] for i, nu in enumerate(self.num_units[:-1])]):
                lstm_type = 'cudnn'
            else:
                lstm_type = 'cudnn_stacked'
        else:
            lstm_type = 'cell'

        saved_state = get_saved_state_vars(self.num_layers, lstm_type)
        if lstm_type == 'cudnn':
            lstm_output, state = add_cudnn_lstm(x_embedded, saved_state, self.num_layers, self.num_units[0])
        elif lstm_type == 'cudnn_stacked':
            lstm_output, state = add_stacked_cudnn_lstm(x_embedded, saved_state, self.num_layers, self.num_units)
        elif lstm_type == 'cell':
            lstm_output, state = add_cell_lstm(x_embedded, saved_state, self.num_layers, self.num_units)

        logits = tf.matmul(lstm_output, self.softmax_layer_matrix) + self.softmax_layer_bias

        save_list = compose_save_list((saved_state, state))
        with tf.control_dependencies(save_list):
            self.predictions = tf.nn.softmax(logits)
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y,
                logits=logits,
            )

    def add_trainable_vars(self):
        self.input_layer_matrix = tf.Variable(
            tf.truncated_normal(
                [self.vocab_size, self.projection_size],
                stddev=self.init_parameter / (self.vocab_size + self.projection_size) ** 0.5
            )
        )
        self.input_layer_bias = tf.Variable(
            tf.zeros(
                [self.projection_size]
            )
        )
        self.softmax_layer_matrix = tf.Variable(
            tf.truncated_normal(
                [self.num_units[-1], self.vocab_size],
                stddev=self.init_parameter / (self.num_units[-1] + self.vocab_size) ** 0.5
            )
        )
        self.softmax_layer_bias = tf.Variable(
            tf.zeros(
                [self.vocab_size]
            )
        )

    def init_placeholders(self):
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
        return loss

    def __call__(self, x):
        feed_dict = self._build_feed_dict(x)
        y_pred = self.sess.run(self.predictions, feed_dict=feed_dict)
        return y_pred





