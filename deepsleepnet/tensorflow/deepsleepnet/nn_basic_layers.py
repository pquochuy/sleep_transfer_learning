import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import tensorflow.contrib.layers as layers

"""
Predefine all necessary layers
"""
# convolution
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        conv = convolve(x, weights)
        #print(conv.get_shape())

        # Add biases
        bias = tf.nn.bias_add(conv, biases)
        bias = tf.reshape(bias, tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)
        return relu

# convolution with batch norm
def conv_bn(x, filter_height, filter_width, num_filters, stride_y, stride_x, is_training, name, padding='SAME'):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        conv = convolve(x, weights)
        #print(conv.get_shape())

        # Add biases
        bias = tf.nn.bias_add(conv, biases)
        bias = tf.reshape(bias, tf.shape(conv))

        conv_bn = tf.contrib.layers.batch_norm(bias, center=True, scale=True,
                                                is_training=is_training, scope=scope)

        # Apply relu function
        relu = tf.nn.relu(conv_bn, name=scope.name)
        return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out])
        biases = tf.get_variable('biases', [num_out])

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)  # Apply ReLu non linearity
            return relu
        else:
            return act

# fully connected layer with batch norm
def fc_bn(x, num_in, num_out, is_training, name, relu=True):
    #with tf.device('/gpu:0'), tf.variable_scope(name) as scope:
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out])
        biases = tf.get_variable('biases', [num_out])

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        act_bn = tf.contrib.layers.batch_norm(act,
                                      center=True, scale=True,
                                      is_training=is_training,
                                      scope='bn')

        if relu == True:
            relu = tf.nn.relu(act_bn)  # Apply ReLu non linearity
            return relu
        else:
            return act_bn


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)



def bidirectional_recurrent_layer(nhidden, nlayer, input_keep_prob=1.0, output_keep_prob=1.0):
    if (nlayer == 1):
        fw_cell = tf.contrib.rnn.LSTMCell(num_units=nhidden,use_peepholes=True,state_is_tuple=True)
        bw_cell = tf.contrib.rnn.LSTMCell(num_units=nhidden,use_peepholes=True,state_is_tuple=True)
    else:
        fw_cell_ = []
        bw_cell_ = []
        for i in range(nlayer):
            fw_cell_.append(tf.contrib.rnn.LSTMCell(num_units=nhidden,use_peepholes=True,state_is_tuple=True))
            bw_cell_.append(tf.contrib.rnn.LSTMCell(num_units=nhidden,use_peepholes=True,state_is_tuple=True))
        fw_cell = tf.contrib.rnn.MultiRNNCell(cells=fw_cell_, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell(cells=bw_cell_, state_is_tuple=True)

    # input & output dropout
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob=input_keep_prob)
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=output_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, input_keep_prob=input_keep_prob)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=output_keep_prob)
    return fw_cell,bw_cell

def bidirectional_recurrent_layer_output(fw_cell, bw_cell, input_layer, sequence_len, scope=None):
    ((fw_outputs,
      bw_outputs),
     (fw_state,
      bw_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                    cell_bw=bw_cell,
                                                    inputs=input_layer,
                                                    sequence_length=sequence_len,
                                                    dtype=tf.float32,
                                                    swap_memory=True,
                                                    scope=scope))
    outputs = tf.concat((fw_outputs, bw_outputs), 2)

    def concatenate_state(fw_state, bw_state):
        if isinstance(fw_state, LSTMStateTuple):
            state_c = tf.concat((fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
            state_h = tf.concat((fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
            state = LSTMStateTuple(c=state_c, h=state_h)
            return state
        elif isinstance(fw_state, tf.Tensor):
            state = tf.concat((fw_state, bw_state), 1,
                              name='bidirectional_concat')
            return state
        elif (isinstance(fw_state, tuple) and
                  isinstance(bw_state, tuple) and
                      len(fw_state) == len(bw_state)):
            # multilayer
            state = tuple(concatenate_state(fw, bw)
                          for fw, bw in zip(fw_state, bw_state))
            return state

        else:
            raise ValueError(
                'unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)
    return outputs, state
