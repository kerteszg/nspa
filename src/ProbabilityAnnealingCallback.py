import numpy as np
from keras.callbacks import Callback

class ProbabilityAnnealingCallback(Callback):
    def __init__(self, loader, epoch_step = 10):
        super(ProbabilityAnnealingCallback, self).__init__()
        self.loader = loader
        self.epoch_step = epoch_step

    def LinearAnnealingProbabilities(self, actual, sh_step=0.1, h_step=0.01, h_max=0.5):
        v = np.zeros(3)
        v[1:] = actual[1:] + [sh_step, h_step]
        if v[2] > h_max:
            v[2] = h_max
        v[0] = 1 - np.sum(v[1:])
        v[v > 1] = 1
        v[v < 0] = 0
        if v.sum() > 1:
            v[v.argmax()] = v[v.argmax()] - (v.sum() - 1)
        return v

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_step != 0:
            return
        actual = self.loader.mixed_probabilities
        new = self.LinearAnnealingProbabilities(actual)
        self.loader.mixed_probabilities = new
        print("Updated probabilities to {} from {}".format(new, actual))

