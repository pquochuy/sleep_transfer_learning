import tensorflow as tf

from nn_basic_layers import *

import numpy as np
import os
from deepsleepnet_config import Config

class DeepSleepNet(object):
    """
    End-to-End DeepSleepNet
    """

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config
        self.input_x = tf.placeholder(tf.float32,shape=[None, self.config.epoch_step, self.config.ntime, self.config.ndim, self.config.nchannel],name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.epoch_step, self.config.nclass], name='input_y')
        self.dropout = tf.placeholder(tf.float32, name="dropout")

        self.istraining = tf.placeholder(tf.bool, name='istraining') # idicate training for batch normmalization

        # store the number of PSG epochs in each input sequence
        # required by SPB's biRNN which is dynamic biRNN
        self.epoch_seq_len = tf.placeholder(tf.int32, [None])

        # fold up the data for epoch processing
        X = tf.reshape(self.input_x, [-1, self.config.ntime, self.config.ndim, self.config.nchannel])

        with tf.device('/gpu:0'), tf.variable_scope("cnn_layers") as scope:
            # CNN brach 1
            conv11 = conv_bn(X, 50, 3, 64, 6, 1, is_training=self.istraining, padding='SAME', name='conv11') # 50 = Fs/2, stride = 6 (according to original implementation)
            print(conv11.get_shape()) # [batchsize x epoch_step, x, 1, 64] (x = (3000-50)/6 + 1 = 492
            pool11 = max_pool(conv11, 8, 1, 8, 1,padding='SAME', name='pool11')
            print(pool11.get_shape())
            # pool1 shape [batchsize x epoch_step, 63, 1, 64] (63 is due to SAME padding above)
            dropout11 = dropout(pool11, self.dropout)

            conv12 = conv_bn(dropout11, 8, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='conv12')
            print(conv12.get_shape()) # [batchsize x epoch_step, 63, 1, 128]
            conv13 = conv_bn(conv12, 8, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='conv13')
            print(conv13.get_shape()) # [batchsize x epoch_step, 63, 1, 128]
            conv14 = conv_bn(conv13, 8, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='conv14')
            print(conv14.get_shape()) # [batchsize x epoch_step, 63, 1, 128]
            pool14 = max_pool(conv14, 4, 1, 4, 1,padding='SAME', name='pool14')
            print(pool14.get_shape()) # [batchsize x epoch_step, 16, 1, 128]
            pool14 = tf.squeeze(pool14) #[batchsize x epoch_step, 16, 128]

            # CNN branch2
            conv21 = conv_bn(X, 400, 3, 64, 50, 1, is_training=self.istraining, padding='SAME', name='conv21') # 400 = Fsx4, 50 = Fs/2
            print(conv21.get_shape()) # [batchsize x epoch_step, x, 1, 64] (x = (3000-400)/50 + 1 = 53
            pool21 = max_pool(conv21, 4, 1, 4, 1,padding='SAME', name='pool21')
            print(pool21.get_shape()) # [batchsize x epoch_step, 15, 1, 64] # 14 is due to SAME padding above
            dropout21 = dropout(pool21, self.dropout)

            conv22 = conv_bn(dropout21, 6, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='conv22')
            print(conv22.get_shape()) # [batchsize x epoch_step, 15, 1, 128]
            conv23 = conv_bn(conv22, 6, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='conv23')
            print(conv23.get_shape()) # [batchsize x epoch_step, 15, 1, 128]
            conv24 = conv_bn(conv23, 6, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='conv24')
            print(conv24.get_shape()) # [batchsize x epoch_step, 15, 1, 128]
            pool24 = max_pool(conv24, 2, 1, 2, 1,padding='SAME', name='pool14')
            print(pool24.get_shape()) # [batchsize x epoch_step, 8, 1, 128]
            pool24 = tf.squeeze(pool24) # [batchsize x epoch_step, 8, 128]

            # concatenate
            cnn_concat = tf.concat([pool14, pool24], axis = 1) # [batchsize x epoch_step, 22, 128]

        with tf.device('/gpu:0'), tf.variable_scope("residual_layer") as scope:
            cnn_output = tf.reshape(cnn_concat, [-1, 24*128]) #[batchsize x epoch_step, 32*128]
            cnn_output = dropout(cnn_output, self.dropout)

            # residual
            residual_output = fc_bn(cnn_output, 24*128, 1024, is_training=self.istraining, name='residual_layer', relu=True)
            #[batchsize x epoch_step, 1024]
            residual_output = tf.reshape(residual_output, [-1, self.config.epoch_step, 1024]) #[batchsize x epoch_step, 32*128]
            print(residual_output.get_shape())

            # unfold data for sequence processing
            rnn_input = tf.reshape(cnn_concat, [-1, self.config.epoch_step, 24*128])

        # bidirectional frame-level recurrent layer
        with tf.device('/gpu:0'), tf.variable_scope("epoch_rnn_layer") as scope:
            fw_cell, bw_cell = bidirectional_recurrent_layer(self.config.nhidden,
                                                                  self.config.nlayer,
                                                                  input_keep_prob=self.dropout,
                                                                  output_keep_prob=self.dropout)
            rnn_out, rnn_state = bidirectional_recurrent_layer_output(fw_cell,
                                                                      bw_cell,
                                                                      rnn_input,
                                                                      self.epoch_seq_len,
                                                                      scope=scope)
            print(rnn_out.get_shape())

            # joint for final output
            final_output = tf.add(rnn_out, residual_output)
            final_output = dropout(final_output, self.dropout)

        # output layer
        self.scores = []
        self.predictions = []
        with tf.device('/gpu:0'), tf.variable_scope("output_layer"):
            for i in range(self.config.epoch_step):
                score_i = fc(tf.squeeze(final_output[:,i,:]),
                                self.config.nhidden * 2,
                                self.config.nclass,
                                name="output", # same variable scope to enforce weight sharing
                                relu=False)
                pred_i = tf.argmax(score_i, 1, name="pred-%s" % i)
                self.scores.append(score_i)
                self.predictions.append(pred_i)

        # calculate cross-entropy output loss
        self.output_loss = 0
        with tf.device('/gpu:0'), tf.name_scope("output-loss"):
            for i in range(self.config.epoch_step):
                output_loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y[:,i,:]), logits=self.scores[i])
                output_loss_i = tf.reduce_sum(output_loss_i, axis=[0])
                self.output_loss += output_loss_i
        self.output_loss = self.output_loss/self.config.epoch_step

        # add on regularization
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss

        self.accuracy = []
        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy"):
            for i in range(self.config.epoch_step):
                correct_prediction_i = tf.equal(self.predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                accuracy_i = tf.reduce_mean(tf.cast(correct_prediction_i, "float"), name="accuracy-%s" % i)
                self.accuracy.append(accuracy_i)

