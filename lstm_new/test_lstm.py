import tensorflow as tf
import numpy as np


class TextLSTM(object):
    """
    A LSTM for text classification.
    Uses an embedding layer, followed by a RNN, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, n_hidden, batch_size, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # expand embedding matrix to 2d matrix
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # reshape embedding matrix tp 3d matrix
            # self.embedded_chars_expanded = tf.reshape(self.embedded_chars_expanded, [-1, embedding_size, n_hidden])

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("cnn-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            # 將h_drop轉換為3維以供lstm使用  ## 384
            self.input = tf.reshape(h_drop, [-1, num_filters_total, 1])

        # 以cnn的結果作為lstm的輸入 _step, n_input

        with tf.name_scope("lstm"):
            W = tf.Variable(tf.random_normal([n_hidden, num_classes]))
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # lstm layer pooled feature
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden, forget_bias=0.1, state_is_tuple=True)
            init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
            outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, self.input, initial_state=init_state, time_major=False)
            self.results = tf.matmul(final_state[1], W) + b

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.predictions = tf.argmax(self.results, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.results, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        print('loading complete')