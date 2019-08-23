import os
#os.environ["CUDA_VISIBLE_DEVICES"]="7,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py

from deepsleepnet import DeepSleepNet
from deepsleepnet_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

from scipy.io import loadmat, savemat


# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("seq_len", 20, "Sequence length (default: 32)")
tf.app.flags.DEFINE_integer("nhidden", 512, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer("evaluate_every", 100, "Numer of training step to evaluate (default: 100)")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

config = Config()
config.dropout = FLAGS.dropout
config.epoch_seq_len = FLAGS.seq_len
config.epoch_step = FLAGS.seq_len
config.nhidden = FLAGS.nhidden
config.evaluate_every = FLAGS.evaluate_every

eeg_active = ((FLAGS.eeg_train_data != "") and (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") and (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") and (FLAGS.emg_test_data != ""))

# EEG data loader and generator
if (eeg_active):
    print("eeg active")
    # Initalize the data generator seperately for the training and test sets
    # actually we do not need to load training data as no normalization (i was lazy to remove)
    eeg_train_gen = DataGenerator(os.path.abspath(FLAGS.eeg_train_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    # no need for data normalization
    eeg_train_gen.X = np.expand_dims(eeg_train_gen.X, axis=-1) # expand feature dimension
    eeg_test_gen.X = np.expand_dims(eeg_test_gen.X, axis=-1) # expand feature dimension

# EOG data loader and generator
if (eog_active):
    print("eog active")
    # Initalize the data generator seperately for the training, validation, and test sets
    eog_train_gen = DataGenerator(os.path.abspath(FLAGS.eog_train_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)

    eog_train_gen.X = np.expand_dims(eog_train_gen.X, axis=-1) # expand feature dimension
    eog_test_gen.X = np.expand_dims(eog_test_gen.X, axis=-1) # expand feature dimension

# EMG data loader and generator
if (emg_active):
    print("emg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    emg_train_gen = DataGenerator(os.path.abspath(FLAGS.emg_train_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)

    emg_train_gen.X = np.expand_dims(emg_train_gen.X, axis=-1) # expand feature dimension
    emg_test_gen.X = np.expand_dims(emg_test_gen.X, axis=-1) # expand feature dimension

# eeg always active
train_generator = eeg_train_gen
test_generator = eeg_test_gen

# expand channel dimension if single channel EEG
if (not(eog_active) and not(emg_active)):
    train_generator.X = np.expand_dims(train_generator.X, axis=-1) # expand channel dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.expand_dims(test_generator.X, axis=-1) # expand channel dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    nchannel = 1
    print(train_generator.X.shape)

# stack in channel dimension if 2 channel EEG+EOG
if (eog_active and not(emg_active)):
    print(train_generator.X.shape)
    print(eog_train_gen.X.shape)
    train_generator.X = np.stack((train_generator.X, eog_train_gen.X), axis=-1) # merge and make new dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.stack((test_generator.X, eog_test_gen.X), axis=-1) # merge and make new dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    nchannel = 2
    print(train_generator.X.shape)

# stack in channel dimension if 2 channel EEG+EOG+EMG
if (eog_active and emg_active):
    print(train_generator.X.shape)
    print(eog_train_gen.X.shape)
    print(emg_train_gen.X.shape)
    train_generator.X = np.stack((train_generator.X, eog_train_gen.X, emg_train_gen.X), axis=-1) # merge and make new dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.stack((test_generator.X, eog_test_gen.X, emg_test_gen.X), axis=-1) # merge and make new dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    nchannel = 3
    print(train_generator.X.shape)

config.nchannel = nchannel

del eeg_train_gen
del eeg_test_gen
if (eog_active):
    del eog_train_gen
    del eog_test_gen
if (emg_active):
    del emg_train_gen
    del emg_test_gen

# shuffle training data here
del train_generator
test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.uint32)

print("Test set: {:d}".format(test_generator.data_size))
print("/Test batches per epoch: {:d}".format(test_batches_per_epoch))


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        han = DeepSleepNet(config=config)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(han.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)



        best_dir = os.path.join(checkpoint_path, "best_model_acc")

        variables = list()
        # only load variables of these scopes
        variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print("RESTORE VARIABLES")
        #print(variables)
        for i, v in enumerate(variables):
            print(v.name[:-2])

        vars_in_checkpoint = tf.train.list_variables(best_dir)
        print("IN-CHECK-POINT VARIABLES")
        #print(vars_in_checkpoint)
        vars_in_checkpoint_names = list()
        for i, v in enumerate(vars_in_checkpoint):
            print(v[0])
            vars_in_checkpoint_names.append(v[0])

        var_list_to_retstore = [v for v in variables if v.name[:-2] in vars_in_checkpoint_names]
        print("ACTUAL RESTORE VARIABLES")
        print(var_list_to_retstore)


        restorer = tf.train.Saver(var_list_to_retstore)

        restorer.restore(sess, best_dir)
        print("Model loaded")

        def dev_step(x_batch, y_batch):
            '''
            A single evaluation step
            '''
            epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
                han.input_x: x_batch,
                han.input_y: y_batch,
                han.dropout: 1.0,
                han.epoch_seq_len: epoch_seq_len,
                han.istraining: 0
            }
            output_loss, total_loss, yhat, score = sess.run(
                   [han.output_loss, han.loss, han.predictions, han.scores], feed_dict)
            return output_loss, total_loss, yhat, score

        def evaluate(gen):
            # Validate the model on the entire data in gen
            output_loss =0
            total_loss = 0
            yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            score = np.zeros([config.epoch_seq_len, len(test_generator.data_index), config.nclass])
            # use 10x of minibatch size to speed up
            num_batch_per_epoch = np.floor(len(gen.data_index) / (10*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch(10*config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*10*config.batch_size : test_step*10*config.batch_size] = yhat_[n]
                    score[n, (test_step-1)*10*config.batch_size : test_step*10*config.batch_size,:] = score_[n]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch(config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, y_batch)
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*10*config.batch_size : len(gen.data_index)] = yhat_[n]
                    score[n, (test_step-1)*10*config.batch_size : len(gen.data_index),:] = score_[n]
                output_loss += output_loss_
                total_loss += total_loss_
            yhat = yhat + 1 # convert to counting from 1

            test_acc = np.zeros([config.epoch_seq_len])
            for n in range(config.epoch_seq_len):
                test_acc[n] = accuracy_score(yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing

            return test_acc, yhat, score, output_loss, total_loss



        # evaluation on test data
        test_acc, test_yhat, test_score, test_output_loss, test_total_loss = evaluate(gen=test_generator)
        # save test results
        savemat(os.path.join(out_path, "test_ret.mat"), dict(yhat = test_yhat, acc = test_acc, score = test_score,
                                                             output_loss = test_output_loss,
                                                             total_loss = test_total_loss))
        test_generator.reset_pointer()

