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

import time
from scipy.io import loadmat


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
tf.app.flags.DEFINE_integer("seq_len", 25, "Sequence length (default: 32)")
tf.app.flags.DEFINE_integer("nhidden", 512, "Sequence length (default: 20)")

# 0: All, 1: softmax+SPB, 2: softmax+EPB, 3: softmax
tf.app.flags.DEFINE_integer("finetune_mode", 0, "Finetuning mode")
# pretrained model checkpoint, set to empty if training from scratch
tf.app.flags.DEFINE_string("pretrained_model", "./pretrained_model/model", "Point to the pretrained model")

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
config.n_hidden = FLAGS.nhidden
config.evaluate_every = FLAGS.evaluate_every

pretrained_model_dir = FLAGS.pretrained_model
finetune_mode = FLAGS.finetune_mode


eeg_active = ((FLAGS.eeg_train_data != "") and (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") and (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") and (FLAGS.emg_test_data != ""))

# EEG data loader and generator
if (eeg_active):
    print("eeg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    eeg_train_gen = DataGenerator(os.path.abspath(FLAGS.eeg_train_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    eeg_eval_gen = DataGenerator(os.path.abspath(FLAGS.eeg_eval_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    # no need for data normalization with raw input
    eeg_train_gen.X = np.expand_dims(eeg_train_gen.X, axis=-1) # expand feature dimension
    eeg_test_gen.X = np.expand_dims(eeg_test_gen.X, axis=-1) # expand feature dimension
    eeg_eval_gen.X = np.expand_dims(eeg_eval_gen.X, axis=-1) # expand feature dimension

# EOG data loader and generator
if (eog_active):
    print("eog active")
    # Initalize the data generator seperately for the training, validation, and test sets
    eog_train_gen = DataGenerator(os.path.abspath(FLAGS.eog_train_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    eog_eval_gen = DataGenerator(os.path.abspath(FLAGS.eog_eval_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    # no need for data normalization with raw input
    eog_train_gen.X = np.expand_dims(eog_train_gen.X, axis=-1) # expand feature dimension
    eog_test_gen.X = np.expand_dims(eog_test_gen.X, axis=-1) # expand feature dimension
    eog_eval_gen.X = np.expand_dims(eog_eval_gen.X, axis=-1) # expand feature dimension

# EMG data loader and generator
if (emg_active):
    print("emg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    emg_train_gen = DataGenerator(os.path.abspath(FLAGS.emg_train_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    emg_eval_gen = DataGenerator(os.path.abspath(FLAGS.emg_eval_data), data_shape=[config.ntime], seq_len=config.epoch_seq_len, shuffle = False)
    # no need for data normalization with raw input
    emg_train_gen.X = np.expand_dims(emg_train_gen.X, axis=-1) # expand feature dimension
    emg_test_gen.X = np.expand_dims(emg_test_gen.X, axis=-1) # expand feature dimension
    emg_eval_gen.X = np.expand_dims(emg_eval_gen.X, axis=-1) # expand feature dimension

# eeg always active
train_generator = eeg_train_gen
test_generator = eeg_test_gen
eval_generator = eeg_eval_gen

# expand channel dimension if single channel EEG
if (not(eog_active) and not(emg_active)):
    train_generator.X = np.expand_dims(train_generator.X, axis=-1) # expand channel dimension
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.expand_dims(test_generator.X, axis=-1) # expand channel dimension
    test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.expand_dims(eval_generator.X, axis=-1) # expand channel dimension
    eval_generator.data_shape = eval_generator.X.shape[1:]
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
    eval_generator.X = np.stack((eval_generator.X, eog_eval_gen.X), axis=-1) # merge and make new dimension
    eval_generator.data_shape = eval_generator.X.shape[1:]
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
    eval_generator.X = np.stack((eval_generator.X, eog_eval_gen.X, emg_eval_gen.X), axis=-1) # merge and make new dimension
    eval_generator.data_shape = eval_generator.X.shape[1:]
    nchannel = 3
    print(train_generator.X.shape)

config.nchannel = nchannel

del eeg_train_gen
del eeg_test_gen
del eeg_eval_gen
if (eog_active):
    del eog_train_gen
    del eog_test_gen
    del eog_eval_gen
if (emg_active):
    del emg_train_gen
    del emg_test_gen
    del emg_eval_gen

# shuffle training data here
train_generator.shuffle_data()

train_batches_per_epoch = np.floor(len(train_generator.data_index) / config.batch_size).astype(np.uint32)
eval_batches_per_epoch = np.floor(len(eval_generator.data_index) / config.batch_size).astype(np.uint32)
test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.uint32)

print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(train_generator.data_size, eval_generator.data_size, test_generator.data_size))
print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))

# variable to keep track of best accuracy on validation set for model selection
best_acc = 0.0

# Training
# ==================================================
early_stop_count = 0

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = DeepSleepNet(config=config)

        # for batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)

            if(pretrained_model_dir == ""):
                print('Scratch training ... ')
                grads_and_vars = optimizer.compute_gradients(net.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            else:
                finetune_vars = list()
                if(finetune_mode == 0): # All
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="cnn_layers"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="residual_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_rnn_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="output_layer"))
                elif(finetune_mode == 1): # softmax+SPB
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_rnn_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="output_layer"))
                elif(finetune_mode == 2): # softmax+EPB
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="cnn_layers"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="residual_layer"))
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="output_layer"))
                elif(finetune_mode == 3): # softmax
                    finetune_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="output_layer"))
                else:
                    print('Inappropriate finetuning mode!')

                print('Finetuning ... ')
                print('FINETUNED VARIABLES')
                print(finetune_vars)
                grads_and_vars = optimizer.compute_gradients(net.loss, var_list=finetune_vars)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        # initialize all variables
        sess.run(tf.initialize_all_variables())
        print("Model initialized")

        if(pretrained_model_dir != ""):
            variables = list()
            # only load variables of these scopes
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cnn_layers"))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="residual_layer"))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_rnn_layer"))
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="output_layer"))

            print("RESTORE VARIABLES")
            #print(variables)
            for i, v in enumerate(variables):
                print(v.name[:-2])


            vars_in_checkpoint = tf.train.list_variables(pretrained_model_dir)
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
            restorer.restore(sess, pretrained_model_dir)
            print("Pretrained model loaded")


        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
              net.input_x: x_batch,
              net.input_y: y_batch,
              net.dropout: config.dropout,
              net.epoch_seq_len: epoch_seq_len,
              net.istraining: 1
            }
            _, step, output_loss, total_loss, accuracy = sess.run(
               [train_op, global_step, net.output_loss, net.loss, net.accuracy],
               feed_dict)
            return step, output_loss, total_loss, accuracy

        def dev_step(x_batch, y_batch):
            """
            A single evaluation step
            """
            epoch_seq_len = np.ones(len(x_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
                net.input_x: x_batch,
                net.input_y: y_batch,
                net.dropout: 1.0,
                net.epoch_seq_len: epoch_seq_len,
                net.istraining: 0
            }
            output_loss, total_loss, yhat = sess.run(
                   [net.output_loss, net.loss, net.predictions], feed_dict)
            return output_loss, total_loss, yhat

        def evaluate(gen, log_filename):
            # Validate the model on the entire data in gen variable
            output_loss =0
            total_loss = 0
            yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            # use minibatch size 10x larger than that in training to speed up
            num_batch_per_epoch = np.floor(len(gen.data_index) / (10*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch(10*config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*10*config.batch_size : test_step*10*config.batch_size] = yhat_[n]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch(config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                for n in range(config.epoch_seq_len):
                    yhat[n, (test_step-1)*10*config.batch_size : len(gen.data_index)] = yhat_[n]
                output_loss += output_loss_
                total_loss += total_loss_
            yhat = yhat + 1 # convert to couting from 1

            # log the accuracies at each time index of the input sequence
            acc = 0
            with open(os.path.join(out_dir, log_filename), "a") as text_file:
                text_file.write("{:g} {:g} ".format(output_loss, total_loss))
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    if n == config.epoch_seq_len - 1:
                        text_file.write("{:g} \n".format(acc_n))
                    else:
                        text_file.write("{:g} ".format(acc_n))
                    acc += acc_n
            acc /= config.epoch_seq_len
            return acc, yhat, output_loss, total_loss

        # Loop over number of epochs
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            step = 1
            while step < train_batches_per_epoch:
                # Get a batch
                x_batch, y_batch, label_batch = train_generator.next_batch(config.batch_size)
                train_step_, train_output_loss_, train_total_loss_, train_acc_ = train_step(x_batch, y_batch)
                time_str = datetime.now().isoformat()

                # average acc over the sequence
                acc_ = 0
                for n in range(config.epoch_seq_len):
                    acc_ += train_acc_[n]
                acc_ /= config.epoch_seq_len

                print("{}: step {}, output_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_total_loss_, acc_))
                step += 1

                current_step = tf.train.global_step(sess, global_step)
                if current_step % config.evaluate_every == 0:
                    print("{} Start validation".format(datetime.now()))
                    # Validate the model on validation data
                    eval_acc, eval_yhat, eval_output_loss, eval_total_loss = evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                    # Validate the model on test data
                    test_acc, test_yhat, test_output_loss, test_total_loss = evaluate(gen=test_generator, log_filename="test_result_log.txt")

                    early_stop_count += 1
                    if(eval_acc >= best_acc):
                        early_stop_count = 0 # reset
                        best_acc = eval_acc
                        checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                        save_path = saver.save(sess, checkpoint_name)

                        print("Best model updated")
                        source_file = checkpoint_name
                        dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                        shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                        shutil.copy(source_file + '.index', dest_file + '.index')
                        shutil.copy(source_file + '.meta', dest_file + '.meta')


                    test_generator.reset_pointer()
                    eval_generator.reset_pointer()

                    # early stop after 50 steps without improvement.
                    # only check after 1000 steps to prevent premature stopping
                    if(current_step >= 1000 and early_stop_count >= 50):
                        quit()

            train_generator.reset_pointer()
            train_generator.shuffle()

