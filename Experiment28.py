import copy
import random as r

import numpy as np

import training

params = {}
params['act_type'] = 'linear'
params['data_name'] = 'Heat_Eqn_exp28'  ## FILL IN HERE (from file name)
params['folder_name'] = 'exp28' # UPDATE so goes in own folder

params['relative_loss'] = 1
params['auto_first'] = 1
params['fixed_L'] = 1
params['dist_weights'] = 'dl'

params['num_evals'] = 10  ## CHECK THIS (how many eigenvalues / how many frequencies / what's the low dimension)
l = params['num_evals']
params['num_real'] = params['num_evals'] # since we expect all real eigenvalues
params['num_complex_pairs'] = 0

n = 128  # number of inputs (spatial discretization)
params['val_len_time'] = 50  ## CHECK THIS (number of time steps)
params['train_len_time'] = [50, 250] * 10 # CHECK THIS. Note: * 10 means repeats this pattern 10 times

params['num_shifts'] = 3
params['delta_t'] = 0.0025  ## FILL IN HERE: your time step

params['max_time'] =  12 * 60 * 60 # this means each experiment will run up to 1 hr
params['num_passes_per_file'] = 15 * 6 * 10 * 50 # may be limiting factor
params['num_steps_per_batch'] = 2

# note: number of ICs is not something that gets added to params. It just helps us with some heuristics below
numICs_max = 400  # CHECK THIS (maximum number of initial conditions)
numICs_min = 80  # CHECK THIS (minimum number of initial conditions)

params['data_train_len'] = 20  ## CHECK THIS (number of training data sets)
params['denoising'] = 0


params['L1_lam'] = 0.0

params['min_5min'] = 10 # essentially no checking
params['min_20min'] = 10
params['min_40min'] = 10
params['min_1hr'] = 5
params['min_2hr'] = 2
params['min_3hr'] = 2
params['min_4hr'] = 1.6
params['min_halfway'] = 1.5 # 1 hours

params['Linf_lam'] = 10 ** (-8)
params['L2_lam'] = 10**(-13.5)
w = 10
params['widths'] = [n, w, l, l, w, n]

len_time_min = min(params['val_len_time'], np.min(params['train_len_time']))
len_time_max = max(params['val_len_time'], np.max(params['train_len_time']))

# must be the smallest because otherwise won't be able to make that many steps forward on some of the data
params['num_shifts_middle'] =  len_time_min-1

max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples_min = numICs_min * (len_time_min - max_shifts)
# i.e. 64 is a nice size for a batch unless a file has fewer than 64 examples
params['batch_size'] = min(num_examples_min, 64)

# Need to estimate a number for num_steps_per_file_pass: if too small, training will stop
# before we reached max_time. If too big, we'll have a very large error table in memory with
# rows pre-allocated for steps we will never reach.
num_examples_max = numICs_max * (len_time_max - max_shifts)
steps_to_see_all = num_examples_max / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']


for count in range(200):
    # only randomized part is learning_rate
    params['learning_rate'] = 10**(-r.uniform(2,4))

    training.main_exp(copy.deepcopy(params))
