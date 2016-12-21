from keras import backend as K
from keras.optimizers import Optimizer
import numpy as np

class PtbSGD(Optimizer):
    def __init__(self, lr=1.0, decay=.5, epoch_size=1000,
                 max_epoch=4, **kwargs):
        super(PtbSGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.base_lr = K.variable(lr)
        self.lr = K.variable(lr)
        self.decay = K.variable(decay)
        self.epoch_size = K.variable(epoch_size)
        self.max_epoch = K.variable(max_epoch)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        epoch = self.iterations // self.epoch_size
        decay = K.pow(self.decay, K.switch(epoch - self.max_epoch > 0.,
                                           epoch - self.max_epoch,
                                           K.variable(0.)))
        self.lr = self.base_lr * decay

        self.updates = [(self.iterations, self.iterations + 1.)]
        for p, g in zip(params, grads):
            self.updates.append((p, p - self.lr * g))
        return self.updates

    def get_config(self):
        config = {'base_lr': float(K.get_value(self.base_lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epoch_size': float(K.get_value(self.epoch_size)),
                  'max_epoch': float(K.get_value(self.max_epoch))}
        base_config = super(PtbSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_lr(self):
        return self.lr.eval()