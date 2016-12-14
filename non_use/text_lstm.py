import tensorflow as tf


class TextLstm(object):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells
    :param rnn_layers: list of int or dict
                        * list of int: the steps used to instantiate the `BasicLstmCell` cell
                        * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return model definition
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        return

    def lstm_cells(self, layers):
        if isinstance(layers[0], dict):
            return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                               state_is_tuple=True))]
