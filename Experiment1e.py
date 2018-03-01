import copy
import random as r

import numpy as np

import training

params = {}
params['data_name'] = 'Heat_Eqn_IC_10_BC_periodic'  ## FILL IN HERE (from file name)
params['folder_name'] = 'exp1e'

params['num_real'] = 10  ## CHECK THIS (how many eigenvalues / how many frequencies)
params['num_complex_pairs'] = 0
params['num_evals'] = params['num_real'] + 2 * params['num_complex_pairs']
l = params['num_evals']

n = 40  # number of inputs (spatial discretization)
params['len_time'] = 101  ## CHECK THIS (number of time steps is 40?)
params['delta_t'] = 0.01  ## FILL IN HERE: your time step

params['max_time'] = 4 * 60 * 60  # this means each experiment will run up to 4 hours
params['num_passes_per_file'] = 15 * 6 * 10
params['num_steps_per_batch'] = 2

params['num_shifts'] = 3
params['num_shifts_middle'] = params['len_time'] - 1
max_shifts = max(params['num_shifts'], params['num_shifts_middle'])

numICs = 10  # CHECK THIS (number of initial conditions)
num_examples = numICs * (params['len_time'] - max_shifts)
params['batch_size'] = numICs
steps_to_see_all = num_examples / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']

params['data_train_len'] = 1  ## CHECK THIS (number of training data sets)
params['denoising'] = 0

params['recon_lam'] = .1
params['L1_lam'] = 0.0
params['L2_lam'] = 0.0

params['min_5min'] = 1 # essentially no checking
params['min_20min'] = 1
params['min_40min'] = 1
params['min_1hr'] = .0001
params['min_2hr'] = .00005
params['min_3hr'] = .00001
params['min_halfway'] = 1 # 2 hours

d = 3
w = 100
params['widths'] = [n, w, w, w, l, l, w, w, w, n]

wo = 10
params['hidden_widths_omega'] = [wo,]


for count in range(200):
    params['auto_first'] = r.randint(0,1)
    params['learning_rate'] = 10**(-r.randint(2,5))
    params['Linf_lam'] = 10 ** (-r.randint(6, 10))

    training.main_exp(copy.deepcopy(params))
