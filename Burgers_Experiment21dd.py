# An experiment to try things out
import copy
import random as r

import numpy as np

import training_convnet as training

params = {}
params['data_name'] = 'Burgers_Eqn_exp21'  ## FILL IN HERE (from file name)
params['folder_name'] = 'Burgers_exp21dd'  # UPDATE so goes in own folder
params['restore'] = 0

params['seed_middle'] = 0
params['fix_middle'] = 0   # Can only fix middle layers if you also seed middle

params['relative_loss'] = 1
params['auto_first'] = 0
params['fixed_L'] = 1
params['diag_L'] = 1

params['mu'] = 1.0
params['num_evals'] = 128  ## CHECK THIS (how many eigenvalues / how many frequencies / what's the low dimension)
l = params['num_evals']
params['num_real'] = params['num_evals']  # since we expect all real eigenvalues
params['num_complex_pairs'] = 0

n = 128  # number of inputs (spatial discretization)
params['len_time'] = 51  ## CHECK THIS (number of time steps)

params['num_shifts'] = 3
params['num_shifts_middle'] = 3
params['delta_t'] = 0.002  ## FILL IN HERE: your time step

params['max_time'] =  1 * 20 * 60  # this means each experiment will run up to 1 hr
params['num_passes_per_file'] = 15 * 6 * 10 * 50  # may be limiting factor
params['num_steps_per_batch'] = 2

numICs = 6000  # CHECK THIS (number of initial conditions)

params['data_train_len'] = 20  ## CHECK THIS (number of training data sets)
params['denoising'] = 0

params['min_5min'] = 10**3 # essentially no checking
params['min_20min'] = 10**2
params['min_40min'] = 10
params['min_1hr'] = 10**4
params['min_2hr'] = 10**4
params['min_3hr'] = 10**4
params['min_4hr'] = 10**4
params['min_halfway'] = 10**4  # 1 hours

params['L1_lam'] = 0.0
params['L2_lam'] = 10**(-8)
params['Linf_lam'] = 0.0
params['autoencoder_loss_lam'] = 1.0
params['prediction_loss_lam'] = 1.0
params['linearity_loss_lam'] = 1.0
params['inner_autoencoder_loss_lam'] = 1.0
params['outer_autoencoder_loss_lam'] = 1.0

params['widths'] = []
params['n_inputs'] = 128
params['conv1_filters'] = 32 # Number of filters in first convolutional layer
params['n_middle'] = 21 
params['conv2_filters'] = 16 # Number of filters in first convolutional layer
params['n_outputs'] = 128

max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = numICs * (params['len_time'] - max_shifts)
params['batch_size'] = min(num_examples, 64)
steps_to_see_all = num_examples / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']

for count in range(200):
    # only randomized part is learning_rate
    params['rand_seed'] = r.randint(0,10**(10))
    r.seed(params['rand_seed'])
    params['learning_rate'] = 10**(-r.uniform(3,6))
    print(params['learning_rate'])
    
    training.main_exp(copy.deepcopy(params))
