import numpy as np


# User inputs
data_train_len = 20  # Number of training data files
data_name = 'Burgers_Eqn_exp28'  # Prefix of data files

data_val = np.loadtxt(('./data/%s_val_x.csv' % data_name), delimiter=',', dtype=np.float32)

np.save(('./data/%s_val_x' % data_name), data_val, allow_pickle=False)

for j in np.arange(data_train_len):
	data_train = np.loadtxt(('./data/%s_train%d_x.csv' % (data_name, j+1)), delimiter=',', dtype=np.float32)
	np.save(('./data/%s_train%d_x' % (data_name, j+1)), data_train, allow_pickle=False)
