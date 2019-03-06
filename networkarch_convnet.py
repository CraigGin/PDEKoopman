import numpy as np
import tensorflow as tf
from scipy.linalg import dft
from functools import partial

import helperfns_convnet

def encoder_apply(x, n_inputs, conv1_filters, n_middle, reg_lam, shifts_middle, num_shifts_max, fix_middle, seed_middle):
    partially_encoded_list = []
    encoded_list = []
    num_shifts_middle = len(shifts_middle)
    for j in np.arange(num_shifts_max + 1):
        if j == 0:
            shift = 0
            reuse = False
        else:
            shift = shifts_middle[j - 1]
            reuse = True
        if isinstance(x, (list,)):
            x_shift = x[shift]
        else:
            x_shift = tf.squeeze(x[shift, :, :])
        partially_encoded, encoded = encoder_apply_one_shift(x_shift, n_inputs, conv1_filters, n_middle, reg_lam, reuse, fix_middle, seed_middle)
        partially_encoded_list.append(partially_encoded)
        if j <= num_shifts_middle:
            encoded_list.append(encoded)

    return partially_encoded_list, encoded_list


def encoder_apply_one_shift(x, n_inputs, conv1_filters, n_middle, reg_lam, reuse, fix_middle, seed_middle):
    prev_layer = tf.identity(x)

    he_init = tf.contrib.layers.variance_scaling_initializer()
    my_conv_layer = partial(tf.layers.conv1d, activation=tf.nn.relu, kernel_initializer=he_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_lam), bias_regularizer=None)

    with tf.variable_scope("encoder", reuse=reuse):
        log_uk = tf.log(prev_layer+1, name="log_uk")
        log_uk_reshaped = tf.reshape(log_uk, shape=[-1, n_inputs, 1], name="log_uk_reshaped")
        hidden1_encode = my_conv_layer(log_uk_reshaped, filters=conv1_filters, kernel_size=1, strides=1, padding="VALID",
                            name="hidden1_encode", reuse=reuse)
        a_encode = tf.get_variable(name="a_encode", shape=[conv1_filters], dtype=tf.float32, initializer=he_init, 
                            regularizer=tf.contrib.layers.l2_regularizer(reg_lam))
        hidden1_encode_scaled= a_encode*hidden1_encode
        hidden2_encode = tf.reduce_sum(hidden1_encode_scaled, axis=2, name="hidden2_encode")
        partially_encoded = tf.layers.dense(hidden2_encode, n_inputs, name="v_k", activation=tf.exp, kernel_initializer=he_init,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_lam), bias_regularizer=None, reuse=reuse)

        


        if not seed_middle:
            FT = tf.get_variable("FT", shape=[n_inputs, n_middle], initializer=he_init, trainable=True, dtype=tf.float32)
        else:
            # Fix last matrix of encoder to be reduced DFT
            DFT = dft(n_inputs)
            rDFT = np.real(DFT)
            iDFT = np.imag(DFT)
            combinedDFT = rDFT[0,:]
            for i in xrange(1,n_inputs/2):
                combinedDFT = np.vstack((combinedDFT, rDFT[i,:]))
                combinedDFT = np.vstack((combinedDFT, iDFT[i,:]))
            combinedDFT = np.vstack((combinedDFT, rDFT[n_inputs/2,:]))
            Reduce = np.hstack((np.eye(n_middle),np.zeros((n_middle,n_inputs-n_middle))))
            Reduced_DFT = Reduce.dot(combinedDFT)
            Reduced_DFT = Reduced_DFT.T
            if not fix_middle:
                FT = tf.get_variable("FT", initializer=np.float32(Reduced_DFT), trainable=True, dtype=tf.float32)
            else:
                FT = tf.get_variable("FT", initializer=np.float32(Reduced_DFT), trainable=False, dtype=tf.float32)
    
        encoded = tf.matmul(partially_encoded,FT, name="vk_hat")

    return partially_encoded, encoded

def decoder_apply(x, n_middle, conv2_filters, n_outputs, reg_lam, reuse, fix_middle, seed_middle):
    prev_layer = tf.identity(x)

    with tf.variable_scope("decoder_inner", reuse=reuse):
        
        if not seed_middle:
            IFT = tf.get_variable("IFT", shape=[n_middle, n_outputs], initializer=tf.contrib.layers.variance_scaling_initializer(), 
                                trainable=True, dtype=tf.float32)
        else:
            # Fix first matrix of decoder to be expanded DFT
            DFT = dft(n_outputs)
            rDFT = np.real(DFT)
            iDFT = np.imag(DFT)
            combinedDFT = rDFT[0,:]
            for i in xrange(1,n_outputs/2):
                combinedDFT = np.vstack((combinedDFT, rDFT[i,:]))
                combinedDFT = np.vstack((combinedDFT, iDFT[i,:]))
            combinedDFT = np.vstack((combinedDFT, rDFT[n_outputs/2,:]))
            Reduce = np.hstack((np.eye(n_middle),np.zeros((n_middle,n_outputs-n_middle))))
            Expand = Reduce.T
            inv_DFT = np.linalg.inv(combinedDFT)
            Expand_DFT = inv_DFT.dot(Expand)
            Expand_DFT = Expand_DFT.T
            if not fix_middle:
                IFT = tf.get_variable("IFT", initializer=np.float32(Expand_DFT), trainable=True, dtype=tf.float32)
            else:
                IFT = tf.get_variable("IFT", initializer=np.float32(Expand_DFT), trainable=False, dtype=tf.float32)


        prev_layer = tf.matmul(prev_layer, IFT) 
    output = outer_decoder_apply(prev_layer, conv2_filters, n_outputs, reg_lam, reuse)

    return output

def outer_decoder_apply(x, conv2_filters, n_outputs, reg_lam, reuse):
    prev_layer = tf.identity(x)

    he_init = tf.contrib.layers.variance_scaling_initializer()
    my_conv_layer = partial(tf.layers.conv1d, activation=tf.nn.relu, kernel_initializer=he_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_lam), bias_regularizer=None)

    with tf.variable_scope("decoder_outer", reuse=reuse):
        log_vkplus1 = tf.log(prev_layer, name="log_vkplus1")
        log_vkplus1_reshaped = tf.reshape(log_vkplus1, shape=[-1, n_outputs, 1], name="log_vkplus1_reshaped")
        hidden1_decode = my_conv_layer(log_vkplus1_reshaped, filters=conv2_filters, kernel_size=4, strides=1, padding="SAME",
                                name="hidden1_decode")
        a_decode = tf.get_variable(name="a_decode", shape=[conv2_filters], dtype=tf.float32, initializer=he_init, 
                                regularizer=tf.contrib.layers.l2_regularizer(reg_lam))
        hidden1_decode_scaled= a_decode*hidden1_decode
        hidden2_decode = tf.reduce_sum(hidden1_decode_scaled, axis=2, name="hidden2_decode")
        output = tf.layers.dense(hidden2_decode, n_outputs, name="outputs", activation=None, kernel_initializer=he_init,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_lam), bias_regularizer=None)

    return output
    
def create_koopman_net(params):
    max_shifts_to_stack = helperfns_convnet.num_shifts_in_stack(params)

    n_inputs = params['n_inputs']
    conv1_filters = params['conv1_filters']
    n_middle = params['n_middle']
    conv2_filters = params['conv2_filters']
    n_outputs = params['n_outputs']

    x = tf.placeholder(tf.float32, shape=[max_shifts_to_stack + 1, None, n_inputs], name="x")

    # returns list: encode each shift
    partial_encoded_list, g_list = encoder_apply(x, n_inputs, conv1_filters, n_middle, reg_lam=params['L2_lam'],
                                                shifts_middle=params['shifts_middle'], num_shifts_max=max_shifts_to_stack, 
                                                fix_middle=params['fix_middle'], seed_middle=params['seed_middle'])

    if not params['seed_middle']:
        with tf.variable_scope("dynamics", reuse=False):
            L = tf.get_variable("L", shape=[n_middle, n_middle], initializer=tf.contrib.layers.variance_scaling_initializer(), 
                                trainable=True, dtype=tf.float32)
    else:
        # Fix middle as heat equation
        max_freq = np.divide(n_middle,2)
        kv = np.empty((n_middle,))
        if n_middle % 2 == 0:
            kv[::2] = np.array(range(max_freq))
        else:
            kv[::2] = np.array(range(max_freq+1))
        kv[1::2] = np.array(range(1,max_freq+1))
        dt = params['delta_t']
        if not params['fix_middle']:
            with tf.variable_scope("dynamics", reuse=False):
                L = tf.get_variable("L", initializer=np.float32(np.diag(np.exp(-params['mu']*kv*kv*dt))), trainable=True, dtype=tf.float32)
        else:
            with tf.variable_scope("dynamics", reuse=False):
                mu = tf.get_variable("mu", shape=[1], initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True, dtype=tf.float32)
                L = tf.diag(tf.exp(-mu*kv*kv*dt), name="L")

    y = []
    # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
    encoded_layer = g_list[0]

    y.append(decoder_apply(encoded_layer, n_middle, conv2_filters, n_outputs, reg_lam=params['L2_lam'], reuse=False, 
                            fix_middle=params['fix_middle'], seed_middle=params['seed_middle']))

    reconstructed_x = []
    for j in np.arange(max_shifts_to_stack + 1):
        reconstructed_x.append(decoder_apply(g_list[j], n_middle, conv2_filters, n_outputs, reg_lam=params['L2_lam'], reuse=True, 
                                            fix_middle=params['fix_middle'], seed_middle=params['seed_middle']))

    outer_reconst_x = []
    for j in np.arange(max_shifts_to_stack + 1):
        outer_reconst_x.append(outer_decoder_apply(partial_encoded_list[j], conv2_filters, n_outputs, 
                                reg_lam=params['L2_lam'], reuse=True))

    if not params['autoencoder_only']:
        # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
        advanced_layer = tf.matmul(encoded_layer, L)

        for j in np.arange(max(params['shifts'])):  # loops 0, 1, ...
            # considering penalty on subset of yk+1, yk+2, yk+3, ... yk+20
            if (j + 1) in params['shifts']:
                y.append(decoder_apply(advanced_layer, n_middle, conv2_filters, n_outputs, reg_lam=params['L2_lam'], reuse=True,
                                        fix_middle=params['fix_middle'], seed_middle=params['seed_middle']))

            advanced_layer = tf.matmul(advanced_layer, L)

    if len(y) != (len(params['shifts']) + 1):
        print "messed up looping over shifts! %r" % params['shifts']
        raise ValueError(
            'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')

    return x, y, partial_encoded_list, g_list, reconstructed_x, outer_reconst_x
