import numpy as np
import tensorflow as tf
import datetime
from functools import partial
import os

# Widths of layers
n_inputs = 128
conv1_filters = 32 # Number of filters in first convolutional layer
n_outputs = 128

data_train_len = 1  # Number of training data files
learning_rate = 10**(-5)
n_epochs = 1000  # Number of steps of optimization procedure
batch_size = 100  # Number of training examples in each batch of mini-batch optimization
reg_lam = 10**(-8)  # Constant weight of regularization loss

folder_name = 'CH_exp4m'  # Folder to be created for saved output
data_name = 'ColeHopf_exp4'  # Prefix of data files
exp_suffix = '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
exp_name = data_name + exp_suffix
model_path = ('./%s/%s_model.ckpt' % (folder_name, exp_name))  # Name of checkpoint file to save

# Set up the graph (Construction phase)
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")

he_init = tf.contrib.layers.variance_scaling_initializer()
my_conv_layer = partial(tf.layers.conv1d, activation=tf.nn.relu, kernel_initializer=he_init,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_lam), bias_regularizer=None)

with tf.name_scope("network"):
    log_X = tf.log(X+1, name="log_X")
    log_X_reshaped = tf.reshape(log_X, shape=[-1, n_inputs, 1], name="log_X_reshaped")
    hidden1 = my_conv_layer(log_X_reshaped, filters=conv1_filters, kernel_size=1, strides=1, padding="VALID",
                             name="hidden1")
    hidden2 = tf.layers.max_pooling1d(hidden1, pool_size=2, strides=1, padding="SAME", name="hidden2")
    hidden2_flat = tf.reshape(hidden2, shape=[-1, n_inputs * conv1_filters])
    output = tf.layers.dense(hidden2_flat, n_outputs, name="outputs", activation=tf.exp, kernel_initializer=he_init,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_lam), bias_regularizer=None)

with tf.name_scope("loss"):
    denominator_nonzero = 10 ** (-6)
    loss_denominator = tf.reduce_sum(tf.square(y), axis=1) + denominator_nonzero
    sum_squares = tf.reduce_sum(tf.squared_difference(output, y), axis=1)
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
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            mse_train = mse.eval(feed_dict={X: X_train, y: y_train})
            rmse_train = np.sqrt(mse_train)
            mse_val = mse.eval(feed_dict={X: X_valid, y: y_valid})
            rmse_val = np.sqrt(mse_val)
            print(epoch, "Training error:", rmse_train, "Val error:", rmse_val)
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    mse_train = mse.eval(feed_dict={X: X_train, y: y_train})
    rmse_train = np.sqrt(mse_train)
    mse_val = mse.eval(feed_dict={X: X_valid, y: y_valid})
    rmse_val = np.sqrt(mse_val)
    print(epoch, "Training error:", rmse_train, "Val error:", rmse_val)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    save_path = saver.save(sess, model_path)
