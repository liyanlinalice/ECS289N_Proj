import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

num_train = 300
num_test = 100
review_cursor = 1
review_scores = np.load('./review_matrices/review_scores.npy')
print('review_scores: ')
print(review_scores)

lr = 1e-3
input_size = 30
hidden_size = 32
layer_num = 2
class_num = 5

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

timestep_size = tf.placeholder(tf.int32, [], name='timestep_size')
batch_size = tf.placeholder(tf.int32, [], name='batch_size')
#None tensor auto adaptable ?
X = tf.placeholder(tf.float32, [1, None, input_size])
y = tf.placeholder(tf.float32, [1, class_num], name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def lstm_cell():
  lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
  lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
  return lstm_cell

mlstm_cell = rnn.MultiRNNCell([lstm_cell() for i in range(layer_num)], state_is_tuple=True)

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state_tuple = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
h_state = state_tuple[layer_num - 1][1]

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32, name='W')
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32, name='bias')

logits = tf.matmul(h_state, W) + bias
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1), name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

def get_next_batch():
    global review_cursor
    review_matrix = np.load('./review_matrices/' + str(review_cursor) + '.npy')
    review_matrix = np.stack([review_matrix], axis=0)
    score = np.zeros([1, 5])
    score[0][int(review_scores[review_cursor - 1]) - 1] = 1
    review_cursor += 1
    return [review_matrix, score]

#Training Session
print("Start to Train!")
sess.run(tf.global_variables_initializer())
for i in range(num_train):
    print("Training Iteration %d" % (i + 1))
    batch = get_next_batch()
    sess.run(train_op, feed_dict={X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: 1})

num_correct = 0
for i in range(num_test):
    print("Testing Iteration %d" % (i + 1))
    batch = get_next_batch()
    test_accuracy = sess.run(accuracy, feed_dict={
        X:batch[0], y: batch[1], keep_prob: 1.0, batch_size: 1})
    print(i, " accuracy=", test_accuracy)
    if test_accuracy > 0:
        num_correct += 1
print("###Overal Test Accuracy = ", num_correct / (num_train + num_test))