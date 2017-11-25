import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape)

lr = 1e-3
batch_size = tf.placeholder(tf.int32, [], name='batch_size')

input_size = 28
timestep_size = 28
hidden_size = 64
layer_num = 2
class_num = 10

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num], name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

X = tf.reshape(_X, [-1, 28, 28], name='X')

def lstm_cell():
  lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
  lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
  return lstm_cell

mlstm_cell = rnn.MultiRNNCell([lstm_cell() for i in range(layer_num)], state_is_tuple=True)

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
print(outputs)
print(state)
h_state = outputs[:, -1, :]

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32, name='W')
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32, name='bias')

# y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias, name='y_pre')
# cross_entropy = -tf.reduce_mean(y * tf.log(y_pre), name='cross_entropy')
logits = tf.matmul(h_state, W) + bias
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1), name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

#Training Session
sess.run(tf.global_variables_initializer())
for i in range(200):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i+1) % 200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            _X:batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
        print("Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

# print("Test accuracy %g"% sess.run(accuracy, feed_dict={
#     _X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))

writer = tf.summary.FileWriter("./graphs", sess.graph)
# print(sess.run(h_state, feed_dict={
#     _X: mnist.test.images[:1], y: mnist.test.labels[:1], keep_prob: 1.0, batch_size: 1
# }))
# print('##########')
# state_tuple = sess.run(state, feed_dict={
#     _X: mnist.test.images[:1], y: mnist.test.labels[:1], keep_prob: 1.0, batch_size: 1
# })
# print(state_tuple[0])
# print('##########')
# print(state_tuple[0][0])
# print(state_tuple[0][1])
# print('##########')
# print(state_tuple[1][0])
# print(state_tuple[1][1])
writer.close()