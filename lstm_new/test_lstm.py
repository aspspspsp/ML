import tensorflow as tf
import numpy as np


class TextLSTM(object):
    """
    A LSTM for text classification.
    Uses an embedding layer, followed by a RNN, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, forget_bias,
      embedding_size, n_hidden, batch_size, l2_reg_lambda=1):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int64, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.input_size = batch_size

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        num_steps = 20
        # Embedding layer
        seq_max_len = 20
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            print('input', self.input_x)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob)
            print('self.embedded_chars', self.h_drop)

        with tf.variable_scope("RNN"):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=0.0, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 5, state_is_tuple=True)  # 多层lstm cell 堆叠起来

            self._initial_state = cell.zero_state(batch_size, tf.float32)  # 参数初始化,rnn_cell.RNNCell.zero_state
            outputs = []
            state = self._initial_state  # state 表示 各个batch中的状态
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # cell_out: [batch, hidden_size]
                (cell_output, state) = cell(self.h_drop[:, time_step, :], state)
                outputs.append(cell_output)  # output: shape[batch][embedding_size, embedding_size]

            # 將結果拼接起來[batch_size, emb_dim, emb_dim]
            self.output = tf.reshape(tf.concat(0, outputs), [-1, num_steps * embedding_size])
            print(self.output)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w", [num_steps * embedding_size, num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=tf.float32)
            self.input_y_hat = tf.matmul(self.output, softmax_w) + softmax_b

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.input_y_hat, self.input_y)
            self.loss = tf.reduce_mean(cross_entropy) # + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.input_y_hat, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        print('loading complete')