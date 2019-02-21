import numpy as np
import tensorflow as tf
import datetime
from functools import partial
import os
from scipy.linalg import dft

# Burgers parameters
mu = 1.0
dt = 0.1

# Widths of layers
n_inputs = 128
conv1_filters = 32 # Number of filters in first convolutional layer
n_middle = 21
conv2_filters = 16 # Number of filters in first convolutional layer
n_outputs = 128

data_train_len = 1  # Number of training data files
learning_rate = 10**(-5)
n_epochs = 1000  # Number of steps of optimization procedure
batch_size = 100  # Number of training examples in each batch of mini-batch optimization
reg_lam = 10**(-8)  # Constant weight of regularization loss

folder_name = 'CH_exp5b'  # Folder to be created for saved output
data_name = 'ColeHopf_exp5'  # Prefix of data files
exp_suffix = '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
exp_name = data_name + exp_suffix
model_path = ('./%s/%s_model.ckpt' % (folder_name, exp_name))  # Name of checkpoint file to save

# Set up the graph (Construction phase)
tf.reset_default_graph()

uk = tf.placeholder(tf.float32, shape=(None, n_inputs), name="u_k")
ukplus1 = tf.placeholder(tf.float32, shape=(None, n_outputs), name="u_kplus1")

he_init = tf.contrib.layers.variance_scaling_initializer()
my_conv_layer = partial(tf.layers.conv1d, activation=tf.nn.relu, kernel_initializer=he_init,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_lam), bias_regularizer=None)

with tf.name_scope("encoder"):
    log_uk = tf.log(uk+1, name="log_uk")
    log_uk_reshaped = tf.reshape(log_uk, shape=[-1, n_inputs, 1], name="log_uk_reshaped")
    hidden1_encode = my_conv_layer(log_uk_reshaped, filters=conv1_filters, kernel_size=1, strides=1, padding="VALID",
                             name="hidden1_encode")
    a_encode = tf.get_variable(name="a_encode", shape=[conv1_filters], dtype=tf.float32, initializer=he_init, 
                         regularizer=tf.contrib.layers.l2_regularizer(reg_lam))
    hidden1_encode_scaled= a_encode*hidden1_encode
    hidden2_encode = tf.reduce_sum(hidden1_encode_scaled, axis=2, name="hidden2_encode")
    vk = tf.layers.dense(hidden2_encode, n_inputs, name="v_k", activation=tf.exp, kernel_initializer=he_init,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_lam), bias_regularizer=None)

with tf.name_scope("dynamics"):
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
    FT = tf.Variable(Reduced_DFT, name="FT", trainable=False, dtype=tf.float32)
    vk_hat = tf.matmul(vk,FT, name="vk_hat")

    max_freq = np.divide(n_middle,2)
    kv = np.empty((n_middle,))
    if n_middle % 2 == 0:
        kv[::2] = np.array(range(max_freq))
    else:
        kv[::2] = np.array(range(max_freq+1))
    kv[1::2] = np.array(range(1,max_freq+1))
    L = tf.Variable(np.diag(np.exp(-mu*kv*kv*dt)), name='L', trainable=False, dtype=tf.float32)
    vkplus1_hat = tf.matmul(vk_hat,L, name="vkplus1_hat")

    Expand = Reduce.T
    inv_DFT = np.linalg.inv(combinedDFT)
    Expand_DFT = inv_DFT.dot(Expand)
    Expand_DFT = Expand_DFT.T
    IFT = tf.Variable(Expand_DFT, name="IFT", trainable=False, dtype=tf.float32)
    vkplus1 = tf.matmul(vkplus1_hat,IFT, name="vkplus1")

with tf.name_scope("decoder"):
    log_vkplus1 = tf.log(vkplus1, name="log_vkplus1")
    log_vkplus1_reshaped = tf.reshape(log_vkplus1, shape=[-1, n_inputs, 1], name="log_vkplus1_reshaped")
    hidden1_decode = my_conv_layer(log_vkplus1_reshaped, filters=conv2_filters, kernel_size=4, strides=1, padding="SAME",
                            name="hidden1_decode")
    a_decode = tf.get_variable(name="a_decode", shape=[conv2_filters], dtype=tf.float32, initializer=he_init, 
                         regularizer=tf.contrib.layers.l2_regularizer(reg_lam))
    hidden1_decode_scaled= a_decode*hidden1_decode
    hidden2_decode = tf.reduce_sum(hidden1_decode_scaled, axis=2, name="hidden2_decode")
    output = tf.layers.dense(hidden2_decode, n_outputs, name="outputs", activation=None, kernel_initializer=he_init,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_lam), bias_regularizer=None)

with tf.name_scope("pred_loss"):
    denominator_nonzero = 10 ** (-6)
    loss_denominator = tf.reduce_sum(tf.square(ukplus1), axis=1) + denominator_nonzero
    sum_squares = tf.reduce_sum(tf.squared_difference(output, ukplus1), axis=1)
    dividing = tf.truediv(sum_squares, loss_denominator)
    mse = tf.reduce_mean(tf.truediv(sum_squares, loss_denominator))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.add_n(reg_losses)
    loss = mse + reg_loss

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

file_writer = tf.summary.FileWriter('logs/'+folder_name, tf.get_default_graph())
file_writer.close()

# Create numpy arrays with data
X_train = np.loadtxt(('./data/%s_train1_x.csv' % data_name), delimiter=',', dtype=np.float32)
y_train = np.loadtxt(('./data/%s_train1_y.csv' % data_name), delimiter=',', dtype=np.float32)
for file_num in range(2, data_train_len + 1):
    X_train = np.append(X_train, np.loadtxt(('./data/%s_train%d_x.csv' % (data_name, file_num)), delimiter=',',
                                            dtype=np.float32), 0)
    y_train = np.append(y_train, np.loadtxt(('./data/%s_train%d_y.csv' % (data_name, file_num)), delimiter=',',
                                            dtype=np.float32), 0)

X_valid = np.loadtxt(('./data/%s_val_x.csv' % data_name), delimiter=',', dtype=np.float32)
y_valid = np.loadtxt(('./data/%s_val_y.csv' % data_name), delimiter=',', dtype=np.float32)

# Define function to randomly split training data into batches
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# Execution phase
with tf.Session() as sess:
    init.run()

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            mse_train = mse.eval(feed_dict={uk: X_train, ukplus1: y_train})
            rmse_train = np.sqrt(mse_train)
            mse_val = mse.eval(feed_dict={uk: X_valid, ukplus1: y_valid})
            rmse_val = np.sqrt(mse_val)
            print(epoch, "Training error:", rmse_train, "Val error:", rmse_val)
            save_path = saver.save(sess, model_path)
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={uk: X_batch, ukplus1: y_batch})

    mse_train = mse.eval(feed_dict={uk: X_train, ukplus1: y_train})
    rmse_train = np.sqrt(mse_train)
    mse_val = mse.eval(feed_dict={uk: X_valid, ukplus1: y_valid})
    rmse_val = np.sqrt(mse_val)
    print(epoch, "Training error:", rmse_train, "Val error:", rmse_val)

    save_path = saver.save(sess, model_path)
