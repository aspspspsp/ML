from keras.models import model_from_json
from keras import initializations
from lstm_ptb.optimizer import PtbSGD
from lstm_ptb.config import get_config
from lstm_ptb.ptb_word_lm import uniform
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import click
import lstm_ptb.reader
import pickle
import sys

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

size = 'medium'
architecture = 'checkpoints/prb_word_lm_medium_architecture.json'
weights = 'checkpoints/prb_word_lm_medium_27_132.82246700553142.h5'
length = 20

config = get_config(size)
initializations.init = uniform(config.init_scale)

with open(architecture) as f:
    model = model_from_json(f.read())

model.load_weights(weights)

optimizer = PtbSGD(lr=config.learning_rate,
                       decay=config.lr_decay,
                       clipnorm=config.max_grad_norm,
                       epoch_size=10, # dummy
                       max_epoch=config.max_epoch)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

with open('vocab.bin', 'rb') as f:
    word_to_id = pickle.load(f)
id_to_word = {}
for c, i in word_to_id.items():
    id_to_word[i] = c

def predict(seed_text):
    sys.stdout.write(seed_text + ' ')
    sentence = [word_to_id[word] for word in seed_text.split(' ')]

    for i in range(length):
        preds = model.predict(pad_sequences([sentence] * config.batch_size,
                                           maxlen=config.num_steps))[0]
        next_index = sample(preds, 1.5)
        next_word = id_to_word[next_index]
        sentence = sentence[1:] + next_index

        sys.stdout.write((next_word if next_word != '<eos>' else '.') + ' ')
        sys.stdout.flush()

from ipywidgets import widgets
from IPython.display import display

text = widgets.Text()
display(text)

def handle_submit(sender):
    predict(text.value)

text.on_submit(handle_submit)