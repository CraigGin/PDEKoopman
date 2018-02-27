import numpy as np
import tensorflow as tf

import helperfns


def weight_variable(shape, var_name, distribution='tn', scale=0.1, first_guess=0):
    if distribution == 'tn':
        initial = tf.truncated_normal(shape, stddev=scale, dtype=tf.float64) + first_guess
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'dl':
        # see page 295 of Goodfellow et al's DL book
        # divide by sqrt of m, where m is number of inputs
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'he':
        # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
        # divide by m, where m is number of inputs
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.random_normal(shape, mean=0, stddev=scale, dtype=tf.float64)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float64)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    var_name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=var_name)


def bias_variable(shape, var_name, distribution=''):
    if distribution:
        initial = np.genfromtxt(distribution, delimiter=',', dtype=np.float64)
    else:
        initial = tf.constant(0.0, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, name=var_name)


def encoder(widths, dist_weights, dist_biases, scale, num_shifts_max, first_guess):
    x = tf.placeholder(tf.float64, [num_shifts_max + 1, None, widths[0]])
    x_noisy = tf.placeholder(tf.float64, [num_shifts_max + 1, None, widths[0]])
    # nx1 patch, number of input channels, number of output channels (features)
    # m = number of hidden units

    weights = dict()
    biases = dict()

    for i in np.arange(len(widths) - 1):
        weights['WE%d' % (i + 1)] = weight_variable([widths[i], widths[i + 1]], var_name='WE%d' % (i + 1),
                                                    distribution=dist_weights[i], scale=scale, first_guess=first_guess)
        # TODO: first guess for biases too (and different ones for different weights)
        biases['bE%d' % (i + 1)] = bias_variable([widths[i + 1], ], var_name='bE%d' % (i + 1),
                                                 distribution=dist_biases[i])
    return x, x_noisy, weights, biases


def encoder_apply(x, weights, biases, act_type, batch_flag, phase, out_flag, shifts_middle, keep_prob, name='E',
                  num_encoder_weights=1):
    y = []
    num_shifts_middle = len(shifts_middle)
    for j in np.arange(num_shifts_middle + 1):
        if j == 0:
            shift = 0
        else:
            shift = shifts_middle[j - 1]
        if isinstance(x, (list,)):
            x_shift = x[shift]
        else:
            x_shift = tf.squeeze(x[shift, :, :])
        y.append(
            encoder_apply_one_shift(x_shift, weights, biases, act_type, batch_flag, phase, out_flag, keep_prob, name,
                                    num_encoder_weights))
    return y


def encoder_apply_one_shift(prev_layer, weights, biases, act_type, batch_flag, phase, out_flag, keep_prob, name='E',
                            num_encoder_weights=1):
    for i in np.arange(num_encoder_weights - 1):
        h1 = tf.matmul(prev_layer, weights['W%s%d' % (name, i + 1)]) + biases['b%s%d' % (name, i + 1)]
        if batch_flag:
            h1 = tf.contrib.layers.batch_norm(h1, is_training=phase)
        if act_type == 'sigmoid':
            h1 = tf.sigmoid(h1)
        elif act_type == 'relu':
            h1 = tf.nn.relu(h1)
        elif act_type == 'elu':
            h1 = tf.nn.elu(h1)
        prev_layer = tf.cond(keep_prob < 1.0, lambda: tf.nn.dropout(h1, keep_prob), lambda: h1)

    final = tf.matmul(prev_layer, weights['W%s%d' % (name, num_encoder_weights)]) + biases[
        'b%s%d' % (name, num_encoder_weights)]

    if (not out_flag) and batch_flag:
        final = tf.contrib.layers.batch_norm(final, is_training=phase)

    return final


def decoder(widths, dist_weights, dist_biases, scale, name='D', first_guess=0):
    weights = dict()
    biases = dict()
    for i in np.arange(len(widths) - 1):
        ind = i + 1
        weights['W%s%d' % (name, ind)] = weight_variable([widths[i], widths[i + 1]], var_name='W%s%d' % (name, ind),
                                                         distribution=dist_weights[ind - 1], scale=scale,
                                                         first_guess=first_guess)
        biases['b%s%d' % (name, ind)] = bias_variable([widths[i + 1], ], var_name='b%s%d' % (name, ind),
                                                      distribution=dist_biases[ind - 1])
    return weights, biases


def decoder_apply(prev_layer, weights, biases, act_type, batch_flag, phase, keep_prob, num_decoder_weights):
    for i in np.arange(num_decoder_weights - 1):
        h1 = tf.matmul(prev_layer, weights['WD%d' % (i + 1)]) + biases['bD%d' % (i + 1)]
        if batch_flag:
            h1 = tf.contrib.layers.batch_norm(h1, is_training=phase)
        if act_type == 'sigmoid':
            h1 = tf.sigmoid(h1)
        elif act_type == 'relu':
            h1 = tf.nn.relu(h1)
        elif act_type == 'elu':
            h1 = tf.nn.elu(h1)
        prev_layer = tf.cond(keep_prob < 1.0, lambda: tf.nn.dropout(h1, keep_prob), lambda: h1)

    return tf.matmul(prev_layer, weights['WD%d' % num_decoder_weights]) + biases['bD%d' % num_decoder_weights]


def form_complex_conjugate_block(omegas, mus, delta_t):
    scale = tf.exp(mus * delta_t)
    entry11 = tf.multiply(scale, tf.cos(omegas * delta_t))
    entry12 = tf.multiply(scale, tf.sin(omegas * delta_t))
    row1 = tf.stack([entry11, -entry12], axis=1)  # [None, 2]
    row2 = tf.stack([entry12, entry11], axis=1)  # [None, 2]
    return tf.stack([row1, row2], axis=2)  # [None, 2, 2] put one row below other


def varying_multiply(y, omegas, delta_t, num_real, num_complex_pairs):
    # multiply on the left: y*omegas

    k = y.shape[1]

    output_list = []

    # first, Jordan blocks for each pair of complex conjugate eigenvalues
    for j in np.arange(num_complex_pairs):
        ind = 2 * j
        ystack = tf.stack([y[:, ind:ind + 2], y[:, ind:ind + 2]], axis=2)  # [None, 2, 2]
        L_stack = form_complex_conjugate_block(omegas[:, ind], omegas[:, ind + 1], delta_t)
        elmtwise_prod = tf.multiply(ystack, L_stack)
        output_list.append(tf.reduce_sum(elmtwise_prod, 1))

    # next, diagonal structure for each real eigenvalue
    # faster to not explicitly create stack of diagonal matrices L
    start_real = 2 * num_complex_pairs
    real_part = tf.multiply(y[:, start_real:], tf.exp(omegas[:, start_real:] * delta_t))

    if len(output_list):
        # each element in list output_list is shape [None, 2]
        complex_part = tf.concat(output_list, axis=1)
    else:
        return real_part

    output = tf.concat([complex_part, real_part], axis=1)

    return output


def create_omega_net(phase, keep_prob, params, ycoords):
    # ycoords is [None, 2] or [None, 3], etc. (temp. only handle 2-diml or 3-diml case)

    weights, biases = decoder(params['widths_omega'], dist_weights=params['dist_weights_omega'],
                              dist_biases=params['dist_biases_omega'], scale=params['scale_omega'], name='O',
                              first_guess=params['first_guess_omega'])
    params['num_omega_weights'] = len(weights)
    omegas = omega_net_apply(phase, keep_prob, params, ycoords, weights, biases, params['num_real'],
                             params['num_complex_pairs'])

    return omegas, weights, biases


def omega_net_apply(phase, keep_prob, params, ycoords, weights, biases, num_real, num_complex_pairs):
    input_list = []
    for j in np.arange(num_complex_pairs):
        ind = 2 * j
        input_list.append(tf.reduce_sum(tf.square(ycoords[:, ind:ind + 2]), axis=1, keep_dims=True))

    start_real = 2 * num_complex_pairs
    real_part = ycoords[:, start_real:]

    if len(input_list):
        # each element in list output_list is shape [None, 1]
        complex_part = tf.concat(input_list, axis=1)
        omega_net_input = tf.concat([complex_part, real_part], axis=1)  # [None, _]

    else:
        omega_net_input = real_part

    omegas = encoder_apply_one_shift(omega_net_input, weights, biases, params['act_type'], params['batch_flag'], phase,
                                     out_flag=0, keep_prob=keep_prob, name='O',
                                     num_encoder_weights=params['num_omega_weights'])

    return omegas


def create_koopman_net(phase, keep_prob, params):
    depth = (params['d'] - 4) / 2  # i.e. 10 or 12 -> 3 or 4

    max_shifts_to_stack = helperfns.num_shifts_in_stack(params)

    encoder_widths = params['widths'][0:depth + 2]  # n ... k
    x, x_noisy, weights, biases = encoder(encoder_widths, dist_weights=params['dist_weights'][0:depth + 1],
                                          dist_biases=params['dist_biases'][0:depth + 1], scale=params['scale'],
                                          num_shifts_max=max_shifts_to_stack, first_guess=params['first_guess'])
    params['num_encoder_weights'] = len(weights)
    g_list = encoder_apply(x_noisy, weights, biases, params['act_type'], params['batch_flag'], phase, out_flag=0,
                           shifts_middle=params['shifts_middle'], keep_prob=keep_prob,
                           num_encoder_weights=params['num_encoder_weights'])

    # g_list_omega is list of omegas, one entry for each middle_shift of x (like g_list)
    omegas, weights_omega, biases_omega = create_omega_net(phase, keep_prob, params, g_list[0])
    # params['num_omega_weights'] = len(weights_omega) already done inside create_omega_net
    weights.update(weights_omega)
    biases.update(biases_omega)

    num_widths = len(params['widths'])
    decoder_widths = params['widths'][depth + 2:num_widths]  # k ... n
    weights_decoder, biases_decoder = decoder(decoder_widths, dist_weights=params['dist_weights'][depth + 2:],
                                              dist_biases=params['dist_biases'][depth + 2:], scale=params['scale'])
    weights.update(weights_decoder)
    biases.update(biases_decoder)

    y = []
    # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
    encoded_layer = g_list[0]
    params['num_decoder_weights'] = depth + 1
    y.append(decoder_apply(encoded_layer, weights, biases, params['act_type'], params['batch_flag'], phase, keep_prob,
                           params['num_decoder_weights']))

    # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
    advanced_layer = varying_multiply(encoded_layer, omegas, params['delta_t'], params['num_real'],
                                      params['num_complex_pairs'])

    for j in np.arange(max(params['shifts'])):  # loops 0, 1, ...
        # considering penalty on subset of yk+1, yk+2, yk+3, ... yk+20
        if (j + 1) in params['shifts']:
            y.append(decoder_apply(advanced_layer, weights, biases, params['act_type'], params['batch_flag'], phase,
                                   keep_prob, params['num_decoder_weights']))

        omegas = omega_net_apply(phase, keep_prob, params, advanced_layer, weights, biases, params['num_real'],
                                 params['num_complex_pairs'])
        advanced_layer = varying_multiply(advanced_layer, omegas, params['delta_t'], params['num_real'],
                                          params['num_complex_pairs'])

    if len(y) != (len(params['shifts']) + 1):
        print "messed up looping over shifts! %r" % params['shifts']
        raise ValueError(
            'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')

    return x, x_noisy, y, g_list, weights, biases
