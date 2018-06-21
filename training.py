import os
import time

import numpy as np
import tensorflow as tf

import helperfns
import networkarch as net


def define_loss(x, y, g_list, weights, biases, params, phase, keep_prob):
    # Minimize the mean squared errors.
    # subtraction and squaring element-wise, then average over both dimensions
    # n columns
    # average of each row (across columns), then average the rows
    denominator_nonzero = 10 ** (-5)

    # autoencoder loss
    if params['autoencoder_loss_lam']:
        if params['relative_loss']:
            loss1_denominator = tf.reduce_mean(
                tf.reduce_mean(tf.square(tf.squeeze(x[0, :, :])), 1)) + denominator_nonzero
        else:
            loss1_denominator = tf.to_double(1.0)
        # compare to original x because want y to have noise removed
        mean_squared_error = tf.reduce_mean(tf.reduce_mean(tf.square(y[0] - tf.squeeze(x[0, :, :])), 1))
        loss1 = params['autoencoder_loss_lam'] * tf.truediv(mean_squared_error, loss1_denominator)
    else:
        loss1 = tf.zeros([1, ], dtype=tf.float64)

    # gets dynamics (prediction loss)
    loss2 = tf.zeros([1, ], dtype=tf.float64)
    if params['prediction_loss_lam']:

        for j in np.arange(params['num_shifts']):
            # xk+1, xk+2, xk+3
            shift = params['shifts'][j]
            if params['relative_loss']:
                loss2_denominator = tf.reduce_mean(
                    tf.reduce_mean(tf.square(tf.squeeze(x[shift, :, :])), 1)) + denominator_nonzero
            else:
                loss2_denominator = tf.to_double(1.0)
            loss2 = loss2 + params['prediction_loss_lam'] * tf.truediv(
                tf.reduce_mean(tf.reduce_mean(tf.square(y[j + 1] - tf.squeeze(x[shift, :, :])), 1)), loss2_denominator)
        loss2 = loss2 / params['num_shifts']

    # K linear
    loss3 = tf.zeros([1, ], dtype=tf.float64)
    if params['linearity_loss_lam']:
        count_shifts_middle = 0

        if params['fixed_L']:
            next_step = tf.matmul(g_list[0], weights['L'])
        else:
            omegas = net.omega_net_apply(phase, keep_prob, params, g_list[0], weights, biases)
            next_step = net.varying_multiply(g_list[0], omegas, params['delta_t'], params['num_real'],
                                             params['num_complex_pairs'])

        for j in np.arange(max(params['shifts_middle'])):
            if (j + 1) in params['shifts_middle']:
                # multiply g_list[0] by L (j+1) times
                # next_step = tf.matmul(g_list[0], L_pow)
                if params['relative_loss']:
                    loss3_denominator = tf.reduce_mean(
                        tf.reduce_mean(tf.square(tf.squeeze(g_list[count_shifts_middle + 1])), 1)) + denominator_nonzero
                else:
                    loss3_denominator = tf.to_double(1.0)
                loss3 = loss3 + params['linearity_loss_lam'] * tf.truediv(
                    tf.reduce_mean(tf.reduce_mean(tf.square(next_step - g_list[count_shifts_middle + 1]), 1)),
                    loss3_denominator)
                count_shifts_middle += 1

            if params['fixed_L']:
                next_step = tf.matmul(next_step, weights['L'])
            else:
                omegas = net.omega_net_apply(phase, keep_prob, params, next_step, weights, biases)
                next_step = net.varying_multiply(next_step, omegas, params['delta_t'], params['num_real'],
                                                 params['num_complex_pairs'])
        loss3 = loss3 / params['num_shifts_middle']

    # inf norm on autoencoder error
    if params['relative_loss']:
        Linf1_den = tf.norm(tf.norm(tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf) + denominator_nonzero
        Linf2_den = tf.norm(tf.norm(tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf) + denominator_nonzero
    else:
        Linf1_den = tf.to_double(1.0)
        Linf2_den = tf.to_double(1.0)
    Linf1_penalty = tf.truediv(
        tf.norm(tf.norm(y[0] - tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf), Linf1_den)
    if x.shape[0] > 1:
        Linf2_penalty = tf.truediv(
            tf.norm(tf.norm(y[1] - tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf), Linf2_den)
    else:
        Linf2_penalty = tf.zeros([1, ], dtype=tf.float64)
    loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty)

    loss = loss1 + loss2 + loss3 + loss_Linf

    return loss1, loss2, loss3, loss_Linf, loss


def define_regularization(params, trainable_var, loss, loss1):
    if params['L1_lam']:
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=params['L1_lam'], scope=None)
        # TODO: don't include biases? use weights dict instead?
        loss_L1 = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=trainable_var)
    else:
        loss_L1 = tf.zeros([1, ], dtype=tf.float64)

    # tf.nn.l2_loss returns number
    l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in trainable_var if 'b' not in v.name])
    loss_L2 = params['L2_lam'] * l2_regularizer

    regularized_loss = loss + loss_L1 + loss_L2
    regularized_loss1 = loss1 + loss_L1 + loss_L2

    return loss_L1, loss_L2, regularized_loss, regularized_loss1


def try_net(data_val, params):
    # SET UP NETWORK
    phase = tf.placeholder(tf.bool, name='phase')
    keep_prob = tf.placeholder(tf.float64, shape=[], name='keep_prob')
    x, x_noisy, y, g_list, weights, biases = net.create_koopman_net(phase, keep_prob, params)

    max_shifts_to_stack = helperfns.num_shifts_in_stack(params)

    # DEFINE LOSS FUNCTION
    trainable_var = tf.trainable_variables()
    loss1, loss2, loss3, loss_Linf, loss = define_loss(x, y, g_list, weights, biases, params, phase, keep_prob)
    loss_L1, loss_L2, regularized_loss, regularized_loss1 = define_regularization(params, trainable_var, loss, loss1)
    losses = {'loss1': loss1, 'loss2': loss2, 'loss3': loss3, 'loss_Linf': loss_Linf, 'loss': loss,
              'loss_L1': loss_L1, 'loss_L2': loss_L2, 'regularized_loss': regularized_loss,
              'regularized_loss1': regularized_loss1}

    # CHOOSE OPTIMIZATION ALGORITHM
    optimizer = helperfns.choose_optimizer(params, regularized_loss, trainable_var)
    optimizer_autoencoder = helperfns.choose_optimizer(params, regularized_loss1, trainable_var)

    # LAUNCH GRAPH AND INITIALIZE
    sess = tf.Session()
    saver = tf.train.Saver()

    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.global_variables_initializer()
    sess.run(init)

    csv_path = params['model_path'].replace('model', 'error')
    csv_path = csv_path.replace('ckpt', 'csv')
    print csv_path

    num_saved_per_file_pass = params['num_steps_per_file_pass'] / 20 + 1
    num_saved = np.floor(num_saved_per_file_pass * params['data_train_len'] * params['num_passes_per_file']).astype(int)
    train_val_error = np.zeros([num_saved, 16])
    count = 0
    best_error = 10000

    data_val_tensor = helperfns.stack_data(data_val, max_shifts_to_stack, params['val_len_time'])
    if params['denoising']:
        data_val_tensor_noisy = helperfns.add_noise(data_val_tensor, params['denoising'], params['rel_noise_flag'])
    else:
        data_val_tensor_noisy = data_val_tensor.copy()

    start = time.time()
    finished = 0
    saver.save(sess, params['model_path'])

    # TRAINING
    # loop over training data files
    for f in xrange(params['data_train_len'] * params['num_passes_per_file']):
        if finished:
            break
        file_num = (f % params['data_train_len']) + 1  # 1...data_train_len

        if (params['data_train_len'] > 1) or (f == 0):
            # don't keep reloading data if always same
            data_train = np.loadtxt(('./data/%s_train%d_x.csv' % (params['data_name'], file_num)), delimiter=',',
                                    dtype=np.float64)
            data_train_tensor = helperfns.stack_data(data_train, max_shifts_to_stack, params['train_len_time'][file_num])
            num_examples = data_train_tensor.shape[1]
            num_batches = int(np.floor(num_examples / params['batch_size']))
        ind = np.arange(num_examples)
        np.random.shuffle(ind)
        data_train_tensor = data_train_tensor[:, ind, :]

        # loop over batches in this file
        for step in xrange(params['num_steps_per_batch'] * num_batches):

            if params['batch_size'] < data_train_tensor.shape[1]:
                offset = (step * params['batch_size']) % (num_examples - params['batch_size'])
            else:
                offset = 0
            batch_data_train = data_train_tensor[:, offset:(offset + params['batch_size']), :]
            if params['denoising']:
                batch_data_train_noisy = helperfns.add_noise(batch_data_train, params['denoising'],
                                                             params['rel_noise_flag'])
            else:
                batch_data_train_noisy = batch_data_train.copy()

            feed_dict_train = {x: batch_data_train, x_noisy: batch_data_train_noisy, phase: 1,
                               keep_prob: params['dropout_rate']}
            feed_dict_train_loss = {x: batch_data_train, x_noisy: batch_data_train_noisy, phase: 1, keep_prob: 1.0}
            feed_dict_val = {x: data_val_tensor, x_noisy: data_val_tensor_noisy, phase: 0, keep_prob: 1.0}

            if (not params['been5min']) and params['auto_first']:
                sess.run(optimizer_autoencoder, feed_dict=feed_dict_train)
            else:
                sess.run(optimizer, feed_dict=feed_dict_train)

            if step % 20 == 0:
                # saves time to bunch operations with one run command (per feed_dict)
                train_errors_dict = sess.run(losses, feed_dict=feed_dict_train_loss)
                val_errors_dict = sess.run(losses, feed_dict=feed_dict_val)
                val_error = val_errors_dict['loss']

                if val_error < (best_error - best_error * (10 ** (-5))):
                    best_error = val_error.copy()
                    saver.save(sess, params['model_path'])
                    reg_train_err = train_errors_dict['regularized_loss']
                    reg_val_err = val_errors_dict['regularized_loss']
                    print("New best val error %f (with reg. train err %f and reg. val err %f)" % (
                        best_error, reg_train_err, reg_val_err))

                train_val_error[count, 0] = train_errors_dict['loss']
                train_val_error[count, 1] = val_error
                train_val_error[count, 2] = train_errors_dict['regularized_loss']
                train_val_error[count, 3] = val_errors_dict['regularized_loss']
                train_val_error[count, 4] = train_errors_dict['loss1']
                train_val_error[count, 5] = val_errors_dict['loss1']
                train_val_error[count, 6] = train_errors_dict['loss2']
                train_val_error[count, 7] = val_errors_dict['loss2']
                train_val_error[count, 8] = train_errors_dict['loss3']
                train_val_error[count, 9] = val_errors_dict['loss3']
                train_val_error[count, 10] = train_errors_dict['loss_Linf']
                train_val_error[count, 11] = val_errors_dict['loss_Linf']
                if np.isnan(train_val_error[count, 10]):
                    params['stop_condition'] = 'loss_Linf is nan'
                    finished = 1
                    break
                train_val_error[count, 12] = train_errors_dict['loss_L1']
                train_val_error[count, 13] = val_errors_dict['loss_L1']
                train_val_error[count, 14] = train_errors_dict['loss_L2']
                train_val_error[count, 15] = val_errors_dict['loss_L2']

                if step % 200 == 0:
                    train_val_error_trunc = train_val_error[range(count), :]
                    np.savetxt(csv_path, train_val_error_trunc, delimiter=',')
                finished, save_now = helperfns.check_progress(start, best_error, params)
                if save_now:
                    train_val_error_trunc = train_val_error[range(count), :]
                    helperfns.save_files(sess, saver, csv_path, train_val_error_trunc, params, weights, biases)
                if finished:
                    break
                count = count + 1

            if step > params['num_steps_per_file_pass']:
                params['stop_condition'] = 'reached num_steps_per_file_pass'
                break

    # SAVE RESULTS
    train_val_error = train_val_error[range(count), :]
    print(train_val_error)
    params['time_exp'] = time.time() - start
    saver.restore(sess, params['model_path'])
    helperfns.save_files(sess, saver, csv_path, train_val_error, params, weights, biases)


def main_exp(params):
    helperfns.set_defaults(params)

    if not os.path.exists(params['folder_name']):
        os.makedirs(params['folder_name'])

    # data is num_steps x num_examples x n
    data_val = np.loadtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',', dtype=np.float64)

    try_net(data_val, params)
    tf.reset_default_graph()
