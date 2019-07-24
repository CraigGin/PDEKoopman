# An experiment to try things out
import copy
import random as r

import numpy as np
import training
import tensorflow as tf

import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

def run(params):
 
    params['data_name'] = 'Burgers_Eqn_exp28'  ## FILL IN HERE (from file name)
    params['folder_name'] = 'Burgers_exp28'  # UPDATE so goes in own folder
    params['restore'] = 0 # Restore a previous model

    n = 128  # number of inputs (spatial discretization)
    params['delta_t'] = 0.002  # Time step
    params['len_time'] = 51  # Number of time steps in each trajectory
    params['mu'] = 1.0  # Strength of viscous (diffusion) term in Burgers' - only needed if seeding middle layer
    numICs = 6000  # Number of initial conditions
    params['data_train_len'] = 20  # Number of training data sets

    params['num_evals'] = 21  # how many eigenvalues / how many frequencies / what's the low dimension
    l = params['num_evals']

    params['network_arch'] = 'fully_connected' # Choose 'convnet' or 'fully_connected'

    # If fully-connected layers:
    params['act_type'] = tf.nn.relu
    params['widths'] = [n]*params['num_encoder_weights'] + [l]*2 + [n]*params['num_encoder_weights']
    params['linear_encoder_layers'] = [params['num_encoder_weights']-2, params['num_encoder_weights']-1]  # 0 is relu, 1&2 are linear
    params['linear_decoder_layers'] = [0, params['num_encoder_weights']-1]  # 0 linear, 1 relu, 2 linear

    # If convolutional neural net
    params['n_inputs'] = 128
    params['conv1_filters'] = 32 # Number of filters in encoder convolutional layer
    params['n_middle'] = 21
    params['conv2_filters'] = 16 # Number of filters in decoder convolutional layer
    params['n_outputs'] = 128

    # For either type of architecture
    params['seed_middle'] = 0  # Seed middle three layers with heat equation (Fourier transform, diagonal matrix, inverse FT)
    params['fix_middle'] = 0   # Can only fix middle layers if you also seed middle

    params['relative_loss'] = 1
    params['auto_first'] = 1 # Train autoencoder only for first 5 minutes

    params['num_shifts'] = params['len_time']-1  # For prediction loss
    params['num_shifts_middle'] = params['len_time']-1  # For linearity loss

    params['max_time'] =  1 * 20 * 60  # hours * minutes * seconds
    params['num_passes_per_file'] = 15 * 6 * 10 * 50  # may be limiting factor
    params['num_steps_per_batch'] = 2

    # Stop if loss is greater than given number
    params['min_5min'] = 10**3 
    params['min_20min'] = 10**2
    params['min_40min'] = 10
    params['min_1hr'] = 10
    params['min_2hr'] = 10
    params['min_3hr'] = 10
    params['min_4hr'] = 10
    params['min_halfway'] = 10  

    # Strength of loss terms
    params['autoencoder_loss_lam'] = 1.0
    params['prediction_loss_lam'] = 1.0
    params['linearity_loss_lam'] = 1.0
    params['inner_autoencoder_loss_lam'] = 1.0
    params['outer_autoencoder_loss_lam'] = 1.0

    max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
    num_examples = numICs * (params['len_time'] - max_shifts)
    params['batch_size'] = min(num_examples, 64)
    steps_to_see_all = num_examples / params['batch_size']
    params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']
    
    best_error = training.main_exp(copy.deepcopy(params))
    return best_error

if __name__ == '__main__':
    params = {}
    params['rand_seed'] = r.randint(0,10**(10))
    r.seed(params['rand_seed'])
    params['learning_rate'] = 10**(-r.uniform(3,6))
    params['L1_lam'] = 0.0
    params['L2_lam'] = 10**(-8)
    params['num_encoder_weights'] = 6
    params['initialization'] = 'He' # Choose 'He' or 'identity'
    params['add_identity'] = 1
    params['diag_L'] = 1  # Middle Layer (dynamics) forced to be diagonal 
    params['log_space'] = 0  # 1 if you want to take a logarithm at beginning of encoder/decoder
    best_error = run(params)

