import tensorflow as tf
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
with tf.device('/gpu:0'):  # Replace with device you are interested in
  bytes_in_use = BytesInUse()
  b = tf.Variable(tf.zeros([10000, 10000]))
  a = tf.Variable(tf.zeros([30000, 30000]))

with tf.Session() as sess:
  print(sess.run(bytes_in_use))
  sess.run(tf.global_variables_initializer())
  print(sess.run(bytes_in_use))


