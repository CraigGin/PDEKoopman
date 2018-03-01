import copy
import random as r

import numpy as np

import training

params = {}
params['data_name'] = 'Heat_Eqn_IC_10_BC_periodic'  ## FILL IN HERE (from file name)
params['folder_name'] = 'exp1b'
params['auto_first'] = 1
params['num_real'] = 10  ## CHECK THIS (how many eigenvalues / how many frequencies)
params['num_complex_pairs'] = 0

params['num_passes_per_file'] = 15 * 6 * 10
params['num_steps_per_batch'] = 2
n = 40  # number of inputs (spatial discretization)
params['len_time'] = 101  ## CHECK THIS (number of time steps is 40?)
params['max_time'] = 60 * 60  # this means each experiment will run up to 1 hour
deltat = 0.01  ## FILL IN HERE: your time step
params['delta_t'] = deltat
params['num_evals'] = params['num_real'] + 2 * params['num_complex_pairs']
l = params['num_evals']

params['num_shifts'] = 3
numICs = 10  # CHECK THIS (number of initial conditions)
params['data_train_len'] = 1  ## CHECK THIS (number of training data sets)
params['recon_lam'] = .1

params['denoising'] = 0
params['L1_lam'] = 0.0
params['learning_rate'] = 10 ** (-3)
params['min_5min'] = 0.07
params['min_20min'] = 0.04
params['min_40min'] = 0.02
params['min_halfway'] = 0.03

for count in range(200):
    params['num_shifts_middle'] = params['len_time'] - 1
    max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
    num_examples = numICs * (params['len_time'] - max_shifts)
    params['batch_size'] = 10
    steps_to_see_all = num_examples / params['batch_size']
    params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']
    params['L2_lam'] = 10 ** (-r.randint(13, 15))
    params['Linf_lam'] = 10 ** (-r.randint(6, 10))

    d = r.randint(1, 4)
    if d == 1:
        wopts = np.arange(50, 160, 10)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [n, w, l, l, w, n]
    elif d == 2:
        wopts = np.arange(15, 45, 5)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [n, w, w, l, l, w, w, n]
    elif d == 3:
        wopts = np.arange(15, 30, 5)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [n, w, w, w, l, l, w, w, w, n]
    elif d == 4:
        wopts = np.arange(15, 25, 5)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [n, w, w, w, w, l, l, w, w, w, w, n]

    do = r.randint(1, 4)
    if do == 1:
        wopts = np.arange(20, 110, 10)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo,]
    elif do == 2:
        wopts = np.arange(10, 25, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, wo]
    elif do == 3:
        wopts = np.arange(5, 20, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, wo, wo]
    elif do == 4:
        wopts = np.arange(5, 15, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, wo, wo, wo]

    training.main_exp(copy.deepcopy(params))
