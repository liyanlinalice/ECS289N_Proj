import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from attention import attention
import time
import os

#Define Model
LR = 1e-2
TIMESTEP_SIZE = 100
INPUT_SIZE = 100
HIDDEN_SIZE = 16
LAYER_NUM = 1
CLASS_NUM = 5
ATTENTION_SIZE = 32

config = tf.ConfigProto(
    device_count = {'GPU': 1}
)

want_to_test = True
want_to_train = True

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
batch_size = tf.placeholder(tf.int32, [], name='batch_size')
#None tensor auto adaptable ?
X = tf.placeholder(tf.float32, [None, TIMESTEP_SIZE, INPUT_SIZE])
y = tf.placeholder(tf.float32, [None, CLASS_NUM], name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
seq_len = tf.placeholder(tf.int32, [None])

def lstm_cell():
    lstm_cell = rnn.BasicLSTMCell(num_units=HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
    #lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell

#init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
#mlstm_cell = rnn.MultiRNNCell([lstm_cell() for i in range(layer_num)], state_is_tuple=True)
#outputs, state_tuple = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
fw_lstm_cell = lstm_cell()
bw_lstm_cell = lstm_cell()
outputs, state_tuple =  bi_rnn(
    fw_lstm_cell,
    bw_lstm_cell,
    inputs=X,
    # initial_state_fw=fw_lstm_cell.zero_state(batch_size, dtype=tf.float32),
    # initial_state_bw=bw_lstm_cell.zero_state(batch_size, dtype=tf.float32),
    sequence_length=seq_len,
    dtype=tf.float32)
attention_output, alphas = attention(outputs, ATTENTION_SIZE, return_alphas=True)

#Difference between [:][-1][:] & [:, -1, :] ?
#h_state = tf.reduce_sum(outputs[:, -1::-10, :], 1)
#h_state = tf.reduce_sum(outputs[:, 0::1, :], 1)
#h_state = state_tuple[layer_num - 1][1]
#print('outputs.get_shape() = ', outputs.get_shape())
#print('h_state.get_shape() = ', h_state.get_shape())

# Dropout
drop = tf.nn.dropout(attention_output, keep_prob)

# Fully Connected

W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, CLASS_NUM], stddev=0.1), dtype=tf.float32, name='W')
bias = tf.Variable(tf.constant(0.1, shape=[CLASS_NUM]), dtype=tf.float32, name='bias')

logits = tf.matmul(attention_output, W) + bias
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
loss_avg = tf.reduce_mean(cross_entropy)
train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy, global_step=global_step)

predict_indices = tf.argmax(logits, 1, name='predict_index')
true_index = tf.argmax(y, 1, name='true_index')

correct_prediction = tf.equal(predict_indices, true_index, name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

# Get Input
review_cursor = 1
review_scores = np.load('./review_matrices/review_scores.npy')
review_word_lens = np.load('./review_matrices/review_word_lens.npy')
print('review_scores_len: ', len(review_scores))

# Evaluating Model
def get_next_batch(size):
    global review_cursor
    review_matrix = []
    stars_distribution = np.zeros([size, 5])
    for i in range(size):
        review_matrix.append(np.load('./review_matrices/' + str(review_cursor) + '.npy'))
        stars_distribution[i][review_scores[review_cursor - 1] -1 ] = 1
        review_cursor += 1
    review_matrix = np.stack(review_matrix, axis=0)

    return [review_matrix, stars_distribution, review_word_lens[review_cursor - size : review_cursor]]

with tf.Session(config=config) as sess:
    #Init
    sess.run(tf.global_variables_initializer())

    #Saver
    saver = tf.train.Saver()

    #Writer
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    #Summary
    with tf.name_scope("suammries"):
        tf.summary.scalar("loss", loss_avg)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("histogram_loss", cross_entropy)
        summary_op = tf.summary.merge_all()

    #If already trained | Why the hell we need to use os.path.dirname?
    check_point = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/checkpoint'))
    if (not want_to_train) and check_point and check_point.model_checkpoint_path:
        print("Start to Restore!")
        saver.restore(sess, check_point.model_checkpoint_path)
    else:

        #Training Session
        print("Start to Train!")
        _batch_size = 30
        for i in range(30000 // _batch_size):
            print("Training Iteration %d" % (i + 1))
            batch = get_next_batch(_batch_size)
            if (i+1)%100 == 0:
                saver.save(sess, './checkpoints/LSTM', global_step=global_step)
                test_accuracy = sess.run(
                    accuracy,
                    feed_dict={X: batch[0], y: batch[1], seq_len: batch[2], keep_prob: 1.0, batch_size: _batch_size}
                )
                print("######################Accuracy = %g" % (test_accuracy))

            _, _, summary = sess.run(
                [loss_avg, train_op, summary_op],
                feed_dict={X: batch[0], y: batch[1], seq_len: batch[2], keep_prob: 0.5, batch_size: _batch_size}
            )

            writer.add_summary(summary, global_step=i)

    if want_to_test:
        #Testing Session
        confusion_mtx = np.zeros([5, 5])
        print("Start to Test!")
        review_cursor = 30001
        _batch_size = 10000
        batch = get_next_batch(_batch_size)
        test_accuracy = sess.run(
            accuracy,
            feed_dict={X: batch[0], y: batch[1], seq_len: batch[2], keep_prob: 1.0, batch_size: _batch_size}
        )
        print("============Accuracy on Test Set = %g" % (test_accuracy))


        test_predict_indices = sess.run(
            predict_indices,
            feed_dict={X: batch[0], y: batch[1], seq_len: batch[2], keep_prob: 1.0, batch_size: _batch_size}
        )

        for i in range(_batch_size):
            confusion_mtx[i % 5, test_predict_indices[i]] += 1
        confusion_mtx /= _batch_size
        print("Confusion Matrix:")
        print(confusion_mtx)