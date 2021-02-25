from model import LSTM_net

import custom_filters

from keras.optimizers import SGD, Adam, Adamax, Nadam, RMSprop, TFOptimizer
import tensorflow as tf

from cost_functions import isd, csd

import corpus


def main():
    run_batch_optimizer(loss='mse', lr=0.001)
    run_batch_optimizer(loss='mse', lr=0.01)
    run_batch_optimizer(loss='mse', lr=0.1)
    run_batch_optimizer(loss='kld', lr=0.001)
    run_batch_optimizer(loss='kld', lr=0.01)
    run_batch_optimizer(loss='kld', lr=0.1)
    run_batch_optimizer(loss='isd', lr=0.001)
    run_batch_optimizer(loss='isd', lr=0.01)
    run_batch_optimizer(loss='isd', lr=0.1)
    run_batch_optimizer(loss='csd', lr=0.001)
    run_batch_optimizer(loss='csd', lr=0.01)
    run_batch_optimizer(loss='csd', lr=0.1)

def run_batch_optimizer(loss='mse', lr=0.001):
    optimizer = Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    run_batch_networks(optimizer, lr, optim_name='Adam', loss=loss)

    optimizer = RMSprop(lr, rho=0.9, epsilon=1e-08, decay=0.0)
    run_batch_networks(optimizer, lr, optim_name='RMSprop', loss=loss)

    optimizer = TFOptimizer(tf.train.RMSPropOptimizer(lr, momentum=0.9))
    run_batch_networks(optimizer, lr, optim_name='TFRMSprop', loss=loss)


def run_batch_networks(optimizer, lr, optim_name='SGD', loss='mse'):
    if loss is 'isd':
        loss_fn = isd
    elif loss is 'csd':
        loss_fn = csd
    else:
        loss_fn = loss

    c = corpus.experiment_files_voc_sample
    network = LSTM_net('voc_LSTM_log_'+optim_name+'_'+str(loss)+'_'+str(lr)+'_'
                             custom_filters.separation_layers)
    network.train(optimizer, c, EPOCHS=100, do_log=True, loss=loss_fn)

    network = LSTM_net('voc_LSTM_log_'+optim_name+'_'+str(loss)+'_'+str(lr)+'_D'+str(0)+'_'
                             custom_filters.separation_layers)
    network.DROPOUT = 0.0
    network.train(optimizer, c, EPOCHS=100, do_log=True, loss=loss_fn)

    network = LSTM_net('voc_LSTM_log_'+optim_name+'_'+str(loss)+'_'+str(lr)+'_D'+str(2)+'_'
                             custom_filters.separation_layers)
    network.DROPOUT = 0.2
    network.train(optimizer, c, EPOCHS=100, do_log=True, loss=loss_fn)

    network = LSTM_net('voc_LSTM_log_'+optim_name+'_'+str(loss)+'_'+str(lr)+'_D'+str(7)+'_'
                             custom_filters.separation_layers)
    network.DROPOUT = 0.7
    network.train(optimizer, c, EPOCHS=100, do_log=True, loss=loss_fn)

    network = LSTM_net('voc_LSTM_log_'+optim_name+'_'+str(loss)+'_'+str(lr)+'_RD'+str(0)+'_'
                             custom_filters.separation_layers)
    network.RDROPOUT = 0.0
    network.train(optimizer, c, EPOCHS=100, do_log=True, loss=loss_fn)

    network = LSTM_net('voc_LSTM_log_'+optim_name+'_'+str(loss)+'_'+str(lr)+'_RD'+str(5)+'_'
                             custom_filters.separation_layers)
    network.RDROPOUT = 0.5
    network.train(optimizer, c, EPOCHS=100, do_log=True, loss=loss_fn)

    network = LSTM_net('voc_LSTM_log_'+optim_name+'_'+str(loss)+'_'+str(lr)+'_RD'+str(7)+'_'
                             custom_filters.separation_layers)
    network.RDROPOUT = 0.7
    network.train(optimizer, c, EPOCHS=100, do_log=True, loss=loss_fn)

    network = LSTM_net('voc_LSTM_log_'+optim_name+'_'+str(loss)+'_'+str(lr)+'_'+str(0)+'_'
                             custom_filters.separation_layers)
    network.RDROPOUT = 0.0
    network.DROPOUT = 0.0
    network.train(optimizer, c, EPOCHS=100, do_log=True, loss=loss_fn)


if __name__ == "__main__":
    main()
    #main(False)
