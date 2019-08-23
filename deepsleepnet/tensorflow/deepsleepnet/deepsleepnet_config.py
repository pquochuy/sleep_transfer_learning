# import tensorflow as tf
import numpy as np
import os


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        # Trainging
        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.001
        self.training_epoch = 10
        self.batch_size = 32
        # dropout for fully connected layers
        self.dropout = 0.5

        self.evaluate_every = 100

        self.nlayer = 2
        self.ndim = 1  # frequency dimension
        self.ntime = 3000  # time dimension
        self.nchannel = 1  # channel dimension
        self.nhidden = 512  # size of hidden state vector of biRNN
        self.nstep = 20  # sequence length
        self.nclass = 5  # Final output classes

        self.epoch_seq_len = 20
