import copy
import random as r

import numpy as np

import training

params = {}
params['data_name'] = 'Heat_Eqn_exp10b'  ## FILL IN HERE (from file name)
params['folder_name'] = 'exp10b' # UPDATE so goes in own folder

params['autoencoder_only'] = 1
params['dist_weights'] = 'dl'

params['num_evals'] = 10  ## CHECK THIS (how many eigenvalues / how many frequencies / what's the low dimension)
l = params['num_evals']

n = 128  # number of inputs (spatial discretization)
params['len_time'] = 101  ## CHECK THIS (number of time steps)
params['delta_t'] = 0.01  ## FILL IN HERE: your time step

params['max_time'] = 4 * 60 * 60  # this means each experiment will run up to 4 hours
params['num_passes_per_file'] = 15 * 6 * 10 # but I think it will quit after 5-ish minutes because this number of loops is so low
params['num_steps_per_batch'] = 2

numICs = 10  # CHECK THIS (number of initial conditions)
num_examples = numICs * params['len_time']
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
params['learning_rate'] = .001


for count in xrange(1000):
    params['Linf_lam'] = 10 ** (-r.randint(6, 10))
    w = 64
    params['widths'] = [n, w, l, l, w, n]

    params['L2_lam'] = 10**(-r.uniform(13,14))
    params['model_path'] = "./%s/%s_%s_model.ckpt" % (params['folder_name'], params['data_name'], str(count))

    training.main_exp(copy.deepcopy(params))
