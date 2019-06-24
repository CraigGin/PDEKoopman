import copy
import random as r

import numpy as np

import training_fixmiddle

params = {}
params['act_type'] = 'relu'
params['data_name'] = 'Burgers_Eqn_exp21'  ## FILL IN HERE (from file name)
params['folder_name'] = 'Burgers_exp21b'  # UPDATE so goes in own folder

params['add_identity'] = 0
params['relative_loss'] = 1
params['auto_first'] = 1
params['fixed_L'] = 1
params['diag_L'] = 0
params['dist_weights'] = 'dl'

params['mu'] = 1.0
params['num_evals'] = 128  ## CHECK THIS (how many eigenvalues / how many frequencies / what's the low dimension)
l = params['num_evals']
params['num_real'] = params['num_evals']  # since we expect all real eigenvalues
params['num_complex_pairs'] = 0

n = 128  # number of inputs (spatial discretization)
params['len_time'] = 51  ## CHECK THIS (number of time steps)

params['num_shifts'] = params['len_time']-1
params['delta_t'] = 0.002  ## FILL IN HERE: your time step

params['max_time'] =  48 * 60 * 60  # this means each experiment will run up to 1 hr
params['num_passes_per_file'] = 15 * 6 * 10 * 50  # may be limiting factor
params['num_steps_per_batch'] = 2

numICs = 6000  # CHECK THIS (number of initial conditions)

params['data_train_len'] = 20  ## CHECK THIS (number of training data sets)
params['denoising'] = 0

params['min_5min'] = 10**4 # essentially no checking
params['min_20min'] = 10**4
params['min_40min'] = 10**4
params['min_1hr'] = 10**4
params['min_2hr'] = 10**4
params['min_3hr'] = 10**4
params['min_4hr'] = 10**4
params['min_halfway'] = 10**4  # 1 hours

params['L1_lam'] = 0.0
params['Linf_lam'] = 0.0
params['L2_lam'] = 10**(-8)
params['autoencoder_loss_lam'] = 1.0
params['prediction_loss_lam'] = 1.0
params['linearity_loss_lam'] = 1.0
params['inner_autoencoder_loss_lam'] = 1.0
params['outer_autoencoder_loss_lam'] = 1.0
# default: last encoder layer and last decoder layer are linear (so that they can output negative numbers)
#
# heat equation could just be lin (128 - 10), lin (10 - 10), lin (10 - 128)
#
# here add nonlinear functions to either side
# encoder = A3*relu(A2*relu(A1*x+b1)+b2)+b3
# relu (128 - 128), relu (128 - 128), lin (128 - 128)
#
# linear transformation: T*L*Tinv
# lin T matrix (128 - 10), lin L matrix (10 - 10), Tinv matrix lin (10 - 128),
#
# decoder = A3*relu(A2*relu(A1*y*b1)+b2)+b3
# relu (128 - 128), relu(128 - 128), lin (128 - 128)
#
params['widths'] = [n, n, n, n, n, n, n, n, n, n, n, n, n, n]
params['linear_encoder_layers'] = [4, 5]  # 0 is relu, 1&2 are linear
params['linear_decoder_layers'] = [0, 5]  # 0 linear, 1 relu, 2 linear

params['num_shifts_middle'] = params['len_time']-1

max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = numICs * (params['len_time'] - max_shifts)
params['batch_size'] = min(num_examples, 64)
steps_to_see_all = num_examples / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']


for count in range(200):
    # only randomized part is learning_rate
    params['learning_rate'] = 10**(-r.uniform(3,5))

    training_fixmiddle.main_exp(copy.deepcopy(params))
