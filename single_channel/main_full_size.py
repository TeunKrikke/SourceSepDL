from model import GRU_net, DNN_net, LSTM_net

from keras.optimizers import SGD, Adam, Adamax, Nadam, RMSprop

from cost_functions import isd, csd

import corpus

import custom_filters


def main():
    run_batch_optimizer(loss='mse')
    # run_batch_optimizer(loss='kld')
    # run_batch_optimizer(loss='isd')
    # run_batch_optimizer(loss='csd')


def run_batch_optimizer(loss='mse'):
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    run_batch_networks(optimizer, optim_name='Adam', loss=loss)

    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # run_batch_networks(optimizer, optim_name='RMSprop', loss=loss)


def run_batch_networks(optimizer, optim_name='SGD', loss='mse', skip_one=False):
    if loss is 'isd':
        loss_fn = isd
    elif loss is 'csd':
        loss_fn = csd
    else:
        loss_fn = loss
        for i in range(50, 201, 50):
            print(i)
            network = GRU_net('to_test/voc_GRU_4L_200E' +
                           optim_name + '_'
                           + str(loss) + str(i) + '.hdf5'
                           , custom_filters.separation_layers)
            network.test_network(optimizer, corpus.experiment_files_voc_test,
                                 loss=loss_fn, files=500)

            network = DNN_net('to_test/voc_DNN_4L_200E' +
                           optim_name + '_'
                           + str(loss) + str(i) + '.hdf5'
                           , custom_filters.separation_layers)
            network.test_network(optimizer, corpus.experiment_files_voc_test,
                                 loss=loss_fn, files=500)
            network = LSTM_net('to_test/voc_LSTM_4L_2dense_200E_500-500' +
                           optim_name + '_'
                           + str(loss) + str(i) + '.hdf5'
                           , custom_filters.separation_layers)
            network.test_network(optimizer, corpus.experiment_files_voc_test,
                                 loss=loss_fn, files=500)

            # network = LSTM('to_test/weights'+str(i)+'.hdf5', False)
            # network.test_network(optimizer, corpus.experiment_files_voc,
            #                      loss=loss_fn, files=1000)

        # network = LSTM('to_test/voc_LSTM_500-500_600E' +
        #                optim_name + '_'
        #                + str(loss) + str(50) + '.hdf5'
        #                , False)
        #
        # network.test_network(optimizer, corpus.experiment_files_voc_test,
        #                      loss=loss_fn, files=100)


if __name__ == "__main__":

    main()
    #main(False)
