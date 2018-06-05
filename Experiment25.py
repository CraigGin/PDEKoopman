import copy
import random as r

import numpy as np

import training

params = {}
params['act_type'] = 'linear'
params['data_name'] = 'Heat_Eqn_exp25'  ## FILL IN HERE (from file name)
params['folder_name'] = 'exp25' # UPDATE so goes in own folder

params['relative_loss'] = 1
params['auto_first'] = 1
params['fixed_L'] = 1
params['dist_weights'] = 'dl'

params['num_evals'] = 11  ## CHECK THIS (how many eigenvalues / how many frequencies / what's the low dimension)
l = params['num_evals']
params['num_real'] = params['num_evals'] # since we expect all real eigenvalues
params['num_complex_pairs'] = 0

n = 128  # number of inputs (spatial discretization)
params['len_time'] = 50  ## CHECK THIS (number of time steps)

params['num_shifts'] = 3
params['delta_t'] = 0.0025  ## FILL IN HERE: your time step

params['max_time'] =  4 * 60 * 60 # this means each experiment will run up to 1 hr
params['num_passes_per_file'] = 15 * 6 * 10 * 50 # may be limiting factor
params['num_steps_per_batch'] = 2

numICs = 400  # CHECK THIS (number of initial conditions)

params['data_train_len'] = 20  ## CHECK THIS (number of training data sets)
params['denoising'] = 0


params['L1_lam'] = 0.0

params['min_5min'] = 10 # essentially no checking
params['min_20min'] = 10
params['min_40min'] = 10
params['min_1hr'] = 10
params['min_2hr'] = 10
params['min_3hr'] = 10
params['min_halfway'] = 10 # 1 hours

params['Linf_lam'] = 10 ** (-8)
params['L2_lam'] = 10**(-13.5)
w = 10
params['widths'] = [n, w, l, l, w, n]

params['num_shifts_middle'] =  params['len_time']-1

max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = numICs * (params['len_time'] - max_shifts)
params['batch_size'] = min(num_examples, 64)
steps_to_see_all = num_examples / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']


for count in range(200):
    # only randomized part is learning_rate
    params['learning_rate'] = 10**(-r.uniform(2,4))

    training.main_exp(copy.deepcopy(params))
