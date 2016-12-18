import numpy as np
import re
import itertools
from collections import Counter

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yookim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\d'", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(
        loss_circulation_data_file, kick_data_file,
        stuck_pipe_data_file, other_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates label.
    Return split sentences and labels
    """
    # Load data from files
    loss_circulation_examples = read_sample(loss_circulation_data_file)
    kick_examples = read_sample(kick_data_file)
    stuck_pipe_example = read_sample(stuck_pipe_data_file)
    other_data_example = read_sample(other_data_file)
    # Split by words
    x_text = loss_circulation_examples + kick_examples + stuck_pipe_example + other_data_example
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    loss_circulation_labels = [[0, 0, 0, 1] for _ in loss_circulation_examples]
    kick_examples_labels = [[0, 0, 1, 0] for _ in kick_examples]
    stuck_pipe_example_labels = [[0, 1, 0, 0] for _ in stuck_pipe_example]
    other_data_example_labels = [[1, 0, 0, 0] for _ in other_data_example]
    y = np.concatenate([loss_circulation_labels, kick_examples_labels, stuck_pipe_example_labels, other_data_example_labels], 0)
    return [x_text, y]

def read_sample(example_file):
    examples = list(open(example_file, "r", errors='ignore').readlines())
    examples = [s.strip() for s in examples]
    return examples

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch interator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]