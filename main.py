import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import time
import os

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

    return [review_matrix, stars_distribution]

def test():
    # Testing Session
    global review_cursor
    review_cursor = 90001
    confusion_mtx = np.zeros([5, 5])
    print("Start to Test!")
    _batch_size = 10000
    batch = get_next_batch(_batch_size)
    test_accuracy = sess.run(
        accuracy,
        feed_dict={X: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size}
    )
    print("============Accuracy on Test Set for Epoch %i = %g" % (epoch, test_accuracy))

    test_predict_indices = sess.run(
        predict_indices,
        feed_dict={X: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size}
    )

    for i in range(_batch_size):
        confusion_mtx[i % 5, test_predict_indices[i]] += 1
    confusion_mtx /= _batch_size
    print("Confusion Matrix:")
    print(confusion_mtx)

# Get Input
review_scores = np.load('./review_matrices/review_scores.npy')
print('review_scores_len: ', len(review_scores))

# Define Hyperparameters

#lr_list = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
want_to_train = True
lr = 1e-2
timestep_size = 100
input_size = 100
hidden_size = 16
layer_num = 1
class_num = 5

config = tf.ConfigProto(
    device_count = {'GPU': 1}
)

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
batch_size = tf.placeholder(tf.int32, [], name='batch_size')
# None tensor auto adaptable ?
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
# Difference between [:][-1][:] & [:, -1, :] ?
# h_state = tf.reduce_sum(outputs[:, -1::-10, :], 1)
h_state = tf.reduce_sum(outputs[:, 0::1, :], 1)
# h_state = state_tuple[layer_num - 1][1]

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32, name='W')
bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32, name='bias')

logits = tf.matmul(h_state, W) + bias
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
loss_avg = tf.reduce_mean(cross_entropy)
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)

predict_indices = tf.argmax(logits, 1, name='predict_index')
true_index = tf.argmax(y, 1, name='true_index')

correct_prediction = tf.equal(predict_indices, true_index, name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

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
        test()
    else:
        for epoch in range(1):
            review_cursor = 1
            #Training Session
            print("Start to Train!")
            _batch_size = 30
            for i in range(90000 // _batch_size):
                #print("Epoch %d Training Iteration %d" % (epoch, i + 1))
                batch = get_next_batch(_batch_size)
                if (i+1)%100 == 0:
                    # saver.save(sess, './checkpoints/LSTM', global_step=global_step)
                    test_accuracy = sess.run(
                        accuracy,
                        feed_dict={X: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size}
                    )
                    #print("######################Accuracy for LR %g = %g" % (lr, test_accuracy))

                _, _, summary = sess.run(
                    [loss_avg, train_op, summary_op],
                    feed_dict={X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size}
                )

                writer.add_summary(summary, global_step=i)
            test()