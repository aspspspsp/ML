import tensorflow as tf
import numpy as np


class TextLSTM(object):
    """
    A LSTM for text classification.
    Uses an embedding layer, followed by a RNN, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, n_hidden, batch_size, n_inputs, n_steps, l2_reg_lambda=0.0):


        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int64, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        seq_max_len = 20
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        # Add dropout
        with tf.name_scope("dropout"):
            keep_prob = 1
            self.h_drop = tf.nn.dropout(self.embedded_chars, keep_prob)

        # lstm layer
        with tf.name_scope("lstm"):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_hidden, state_is_tuple=True)
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)
            self.output, self.state = tf.nn.dynamic_rnn(cell, self.h_drop, initial_state=initial_state)

        with tf.name_scope("mean_pooling_layer"):
            self.output = tf.unpack(tf.transpose(self.output, [1, 0, 2]))

        # Final (unnormalized) scores and predictions
        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w", [n_hidden, num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=tf.float32)
            self.predictions = tf.matmul(self.output[-1], softmax_w) + softmax_b

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.predictions, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        print('loading complete')