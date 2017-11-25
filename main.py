import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

review_cursor = 1
review_scores = np.load('./review_matrices/review_scores.npy')
print('review_scores: ')
print(review_scores)

lr = 1e-3
timestep_size = 100
input_size = 200
hidden_size = 16
layer_num = 1
class_num = 5

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

batch_size = tf.placeholder(tf.int32, [], name='batch_size')
#None tensor auto adaptable ?
X = tf.placeholder(tf.float32, [None, timestep_size, input_size])
y = tf.placeholder(tf.float32, [None, class_num], name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def lstm_cell():
  lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
  lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
  return lstm_cell

mlstm_cell = rnn.MultiRNNCell([lstm_cell() for i in range(layer_num)], state_is_tuple=True)

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state_tuple = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
#Difference between [:][-1][:] & [:, -1, :] ?
h_state = tf.reduce_sum(outputs[:, -1::-10, :], 1)
#h_state = state_tuple[layer_num - 1][1]
print('outputs.get_shape() = ', outputs.get_shape())
print('h_state.get_shape() = ', h_state.get_shape())

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32, name='W')
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32, name='bias')

logits = tf.matmul(h_state, W) + bias
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1), name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

def get_next_batch(size):
    global review_cursor
    review_matrix = []
    score = np.zeros([size, 5])
    for i in range(size):
        review_matrix.append(np.load('./review_matrices/' + str(review_cursor) + '.npy'))
        score[i][int(review_scores[review_cursor - 1]) -1 ] = 1
        review_cursor += 1
    review_matrix = np.stack(review_matrix, axis=0)

    return [review_matrix, score]

#Training Session
print("Start to Train!")
sess.run(tf.global_variables_initializer())
_batch_size = 10
for i in range(90000 // _batch_size):
    print("Training Iteration %d" % (i + 1))
    batch = get_next_batch(_batch_size)
    if (i+1)%100 == 0:
        test_accuracy = sess.run(
            accuracy,
            feed_dict={X: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size}
        )
        print("######################Accuracy = %g" % (test_accuracy))

    sess.run(train_op, feed_dict={X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

    # print("^^^^^^^^^^^^^^^shape of h_state")
    # print(sess.run(tf.shape(h_state), feed_dict={X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size}))

#Testing Session
print("Start to Test!")
_batch_size = 800
batch = get_next_batch(_batch_size)
test_accuracy = sess.run(
    accuracy,
    feed_dict={X: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size}
)
print("============Accuracy on Test Set = %g" % (test_accuracy))