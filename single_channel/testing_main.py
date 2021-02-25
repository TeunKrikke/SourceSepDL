from model import CNN, LSTM, LSTM_backwards, DNN, RCNN,LSTM_without, LSTM_ideal, LSTM_tanh, LSTM_linear, LSTM_pobs

from keras.optimizers import SGD, Adam, Adamax, Nadam, RMSprop, Adagrad, TFOptimizer, Adadelta

from cost_functions import isd, csd

import corpus

import tensorflow as tf


def main():
    # optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    # network = LSTM('voc_LSTM_Test', False)
    # network.test_network('voc_LSTM_nadam_mse', optimizer, corpus.experiment_files_voc)
    run_batch_optimizer(loss='mse')
#     run_batch_optimizer(loss='kl')
#     # run_batch_optimizer(loss=isd,s1=s1)
#     # run_batch_optimizer(loss=csd,s1=s1)
#


def run_batch_optimizer(loss='mse'):
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # run_batch_networks(optimizer, optim_name='Adam', loss=loss, skip_one=True)

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    run_batch_networks(optimizer, optim_name='RMSprop', loss=loss)

    optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    run_batch_networks(optimizer, optim_name='nadam', loss=loss)
#
    optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    run_batch_networks(optimizer, optim_name='Adamax', loss=loss)

    # optimizer = Adagrad()
    # run_batch_networks(optimizer, optim_name='Adagrad', loss=loss)
    # optimizer = TFOptimizer(tf.train.MomentumOptimizer(0.001))
    # run_batch_networks(optimizer, optim_name='Momentum', loss=loss)
    # optimizer = Adadelta()
    # run_batch_networks(optimizer, optim_name='Adadelta', loss=loss)






def run_batch_networks(optimizer, optim_name='SGD', loss='mse', skip_one=False):

    network = LSTM('voc_LSTM_log_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_log=True)

    network = LSTM('voc_LSTM_mag_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_mag=True)

    # network = LSTM_pobs('voc_LSTM_pobs_raw_'+optim_name+'_'+str(loss), False)
    # network.test_network(optimizer, corpus.experiment_files_voc, pobs=True)

    network = LSTM_without('voc_LSTM_no_log_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_log=True)

    network = LSTM_without('voc_LSTM_no_mag_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_mag=True)

    network = LSTM_without('voc_LSTM_no_raw_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc)

    network = LSTM_linear('voc_LSTM_lin_log_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_log=True)

    network = LSTM_linear('voc_LSTM_lin_mag_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_mag=True)

    network = LSTM_linear('voc_LSTM_lin_raw_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc)

    network = LSTM_tanh('voc_LSTM_tanh_log_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_log=True)

    network = LSTM_tanh('voc_LSTM_tanh_mag_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_mag=True)

    network = LSTM_tanh('voc_LSTM_tanh_raw_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc)

    network = LSTM_ideal('voc_LSTM_id_log_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_log=True)

    network = LSTM_ideal('voc_LSTM_id_mag_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc, do_mag=True)

    network = LSTM_ideal('voc_LSTM_id_raw_'+optim_name+'_'+str(loss), False)
    network.test_network(optimizer, corpus.experiment_files_voc)

if __name__ == "__main__":
    main()
    #main(False)
