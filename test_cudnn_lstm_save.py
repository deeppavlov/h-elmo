import os
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnLSTM as CudnnLSTM
inp = tf.zeros([10, 32, 100])
lstm1 = CudnnLSTM(1, 128)
lstm2 = CudnnLSTM(2, 256)
lstm1.build(inp.shape)
lstm2.build(inp.shape)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save_path = 'test_cudnn_lstm_save/1'
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path))
    saver.save(sess, save_path)