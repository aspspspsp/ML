'''
A Dynamic Recurrent Neural Network (LSTM) implementation example using
TensorFlow library. This example is using a toy dataset to classify linear
sequences. The generated sequences have variable length.
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from lstm import data_helpers
from tensorflow.contrib import learn
import numpy as np
# ==========
#   MODEL
# ==========

# Hyper parameters
learning_rate = 0.01
training_iters = 1000000 #1000000
batch_size = 128
n_inputs = 10
n_steps = 21
display_step = 10
check_point_every = 1000

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for vaildation")
tf.flags.DEFINE_string("loss_circulation_data_file", "../data/loss_circulation_data.txt", "Data source for loss circulation")
tf.flags.DEFINE_string("kick_data_file", "../data/kick_data.txt", "Data source for kick")
tf.flags.DEFINE_string("stuck_pipe_data_file", "../data/stuck_pipe_data.txt", "Data source for stuck pipe")
tf.flags.DEFINE_string("other_data_file", "../data/other_data.txt", "Data source for other")

FLAGS = tf.flags.FLAGS

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(
    FLAGS.loss_circulation_data_file, FLAGS.kick_data_file,
    FLAGS.stuck_pipe_data_file, FLAGS.other_data_file)

# Network Parameters
n_hidden = 128 # hidden layer num of features
n_classes = y.shape[1] # linear sequence or not

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # shape (30, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden])),
    # shape (128, 4)
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden, ])),
    # shape (4, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def dynamicRNN(X, n_inputs, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 210 steps, 128 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 210 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden])

     # 使用 basic LSTM Cell.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.1, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    return results

pred = dynamicRNN(x, n_inputs, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Generate batches
batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)),
            batch_size, training_iters)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    for batch in batches:
        # Read batch of data
        batch_x, batch_y = zip(*batch)
        # Reshape data
        batch_x = [np.array(batch_x[i]).reshape(n_steps, n_inputs) for i in range(0, len(batch_x))]

        if len(batch_x) != batch_size:
            continue

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        if(step >= training_iters):
            break
        #if step % check_point_every == 0:
        # do something

        step += 1

    print("Optimization Finished!")