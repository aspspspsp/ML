import numpy as np
import os
def load():
        # load text
        loss_cir = read_sample("../data/loss_circulation_data.txt")
        kick = read_sample('../data/loss_circulation_data.txt')
        st_pipe = read_sample('../data/stuck_pipe_data.txt')
        other = read_sample('../data/other_data.txt')

        x = loss_cir + kick + st_pipe + other

        # load label
        loss_cir_labels = [0 for _ in loss_cir]
        kick_labels = [1 for _ in kick]
        st_pipe_labels = [2 for _ in st_pipe]
        other_example_labels = [3 for _ in other]
        y = np.concatenate([loss_cir_labels, kick_labels, st_pipe_labels, other_example_labels], 0)

        return x, y

def read_sample(example_file):
    examples = list(open(example_file, "r", errors='ignore').readlines())
    examples = [s.strip() for s in examples]
    return examples