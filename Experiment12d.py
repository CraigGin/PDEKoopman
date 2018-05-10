import copy
import random as r

import numpy as np

import training

params = {}
params['act_type'] = 'linear'
params['data_name'] = 'Heat_Eqn_exp12'  ## FILL IN HERE (from file name)
params['folder_name'] = 'exp12d' # UPDATE so goes in own folder

params['fixed_L'] = 1
params['dist_weights'] = 'dl'

params['num_evals'] = 10  ## CHECK THIS (how many eigenvalues / how many frequencies / what's the low dimension)
l = params['num_evals']
params['num_real'] = params['num_evals'] # since we expect all real eigenvalues
params['num_complex_pairs'] = 0

n = 128  # number of inputs (spatial discretization)
params['len_time'] = 101  ## CHECK THIS (number of time steps)

params['num_shifts'] = 1
params['num_shifts_middle'] =  1
params['delta_t'] = 0.01  ## FILL IN HERE: your time step

params['max_time'] = 4 * 60 * 60  # this means each experiment will run up to 4 hours
params['num_passes_per_file'] = 15 * 6 * 10 * 50 # this is sometimes the limiting factor
params['num_steps_per_batch'] = 2

numICs = 10  # CHECK THIS (number of initial conditions)

max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = numICs * (params['len_time'] - max_shifts)
params['batch_size'] = num_examples
steps_to_see_all = num_examples / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']


params['data_train_len'] = 1  ## CHECK THIS (number of training data sets)
params['denoising'] = 0


params['L1_lam'] = 0.0

params['min_5min'] = 1 # essentially no checking
params['min_20min'] = 1
params['min_40min'] = 1
params['min_1hr'] = .0001
params['min_2hr'] = .00005
params['min_3hr'] = .00001
params['min_halfway'] = 1 # 2 hours


for count in range(200):
    params['auto_first'] = r.randint(0,1)

    params['Linf_lam'] = 10 ** (-r.randint(6, 10))
    w = 64
    params['widths'] = [n, w, l, l, w, n]

    params['L2_lam'] = 10**(-r.uniform(13,14))

    params['learning_rate'] = 10**(-r.uniform(2,4))

    training.main_exp(copy.deepcopy(params))
