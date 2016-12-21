from keras import backend as K
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from lstm_ptb.optimizer import PtbSGD
from lstm_ptb.config import get_config
import click
import numpy as np
import lstm_ptb.reader
import time
import pickle

def get_model(epoch_size, config):
    """Return the PTB model."""
    batch_size = config.batch_size
    num_steps = config.num_steps
    num_layers = config.num_layers
    size = config.hidden_size
    vocab_size = config.vocab_size
    learning_rate = config.learning_rate
    lr_decay = config.lr_decay
    keep_prob = config.keep_prob
    max_grad_norm = config.max_grad_norm
    max_epoch = config.max_epoch
    max_max_epoch = config.max_max_epoch

    lstm_parameters = {
        "output_dim":size,
        "init":uniform(config.init_scale),
        "inner_init":uniform(config.init_scale),
        "forget_bias_init":"zero",
        "stateful":True,
        "consume_less":"gpu"
    }

    model = Sequential()

    model.add(Embedding(vocab_size, size,
                        batch_input_shape=(batch_size, num_steps)))

    if keep_prob < 1:
        model.add(Dropout(1 - keep_prob))

    for i in range(num_layers - 1):
        model.add(LSTM(return_sequences=True, **lstm_parameters))
        if keep_prob < 1:
            model.add(Dropout(1- keep_prob))

    model.add(LSTM(return_sequences=False, **lstm_parameters))
    if keep_prob < 1:
        model.add(Dropout(1 - keep_prob))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    optimizer = PtbSGD(lr=learning_rate, decay=lr_decay,
                       clipnorm=max_grad_norm,
                       epoch_size=epoch_size,
                       max_epoch=max_epoch)

    # lr 1だとネットワークが大きい場合にあっという間にperplexityが発散して行っちゃうんだけど？
    # optimizer = SGD(lr=learning_rate, clipnorm=max_grad_norm)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def run_epoch(data, model, batch_size, num_steps, vocab_size):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // batch_size) - 1) // num_steps
    start_time = time.time()
    losses = 0.0
    iters = 0

    model.reset_states()

    for step, (x, y) in enumerate(reader.ptb_iterator(data, batch_size, num_steps)):
        y = to_categorical(y, nb_classes=vocab_size)
        loss = model.train_on_batch(x, y)
        losses += loss
        iters += num_steps

        # print(model.optimizer.get_lr())
        print(np.exp(losses / iters))
        if step % (epoch_size // 10) == 10:
            print('{:.3f} perplexity: {:.3f} speed: {:.0f} wps'.format(
                step * 1.0 / epoch_size, np.exp(losses / iters),
                iters * batch_size / (time.time() - start_time)
            ))

    return np.exp(losses / iters)

def run_test_epoch(data, model, batch_size, num_steps, vocab_size):
    """Tests the model on the given data."""
    epoch_size = ((len(data) // batch_size) - 1) // num_steps
    losses = 0.0
    iters = 0

    model.reset_states()

    for step, (x, y) in enumerate(reader.ptb_iterator(data, batch_size, num_steps)):
        y = to_categorical(y, nb_classes=vocab_size)
        loss = model.test_on_batch(x, y)
        losses += loss
        iters += num_steps

    return np.exp(losses / iters)

def uniform(scale=0.05):
    def init(shape, name=None):
        return K.variable(np.random.uniform(low=-scale, high=scale, size=shape),
                          name=name)
    return init

@click.command()
@click.option('--size', default='small')
@click.option('--data_path', default='data/simple-examples/data')
def main(size, data_path):
    raw_data = reader.ptb_raw_data(data_path)
    word_to_id, id_to_word, train_data, valid_data, test_data = raw_data
    config = get_config(size)
    batch_size = config.batch_size
    num_steps = config.num_steps
    vocab_size = config.vocab_size
    epoch_size = ((len(train_data) // config.batch_size) - 1) // config.num_steps
    model = get_model(epoch_size, config)

    with open('vocab.bin', 'wb') as f:
        pickle.dump(word_to_id, f)

    print('Training with {} size'.format(size))

    with open('checkpoints/prb_word_lm_{}_architecture.json'.format(size), 'w') as f:
        f.write(model.to_json())

    # train
    for i in range(config.max_max_epoch):
        # print("Epoch: {} Learning rate: {}".format(i + 1, model.optimizer.get_lr()))
        train_perplexity = run_epoch(train_data, model, batch_size, num_steps, vocab_size)
        print('Epoch: {} Train Perplexity: {:.3f}'.format(
            i + 1, train_perplexity))
        valid_perplexity = run_test_epoch(valid_data, model, batch_size, num_steps, vocab_size)
        print('Epoch: {} Valid Perplexity: {:.3f}'.format(
            i + 1, valid_perplexity))
        print('save weights ...')
        model.save_weights('checkpoints/prb_word_lm_{}_{}_{}.h5'.format(
            size, i, valid_perplexity))

    test_perplexity = run_test_epoch(test_data, model, batch_size, num_steps, vocab_size)
    print('Test Perplexity: {:.3f}'.format(test_perplexity))

if __name__ == '__main__':
    main()