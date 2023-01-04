from model import CNN, LSTM, LSTM_backwards, DNN, RCNN, LSTM_single_speaker, LSTM_backwards_single_speaker, DNN_single_speaker

from keras.optimizers import SGD, Adam, Adamax, Nadam, RMSprop

from cost_functions import isd, csd

import corpus


def main():
    run_batch_optimizer(loss='mse')
    # run_batch_optimizer(loss='kld')
    run_batch_optimizer(loss='isd')
    # run_batch_optimizer(loss='csd')


def run_batch_optimizer(loss='mse'):
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # run_batch_networks(optimizer, optim_name='Adam', loss=loss)

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    run_batch_networks(optimizer, optim_name='RMSprop', loss=loss)


def run_batch_networks(optimizer, optim_name='SGD', loss='mse', skip_one=False):
    if loss is 'isd':
        loss_fn = isd
    elif loss is 'csd':
        loss_fn = csd
    else:
        loss_fn = loss
    if not skip_one:
        for i in range(50, 601, 50):
            print(i)
            network = LSTM('to_test/voc_LSTM_4L_600E' +
                           optim_name + '_'
                           + str(loss) + str(i) + '.hdf5'
                           , False)
            network.do_explain(optimizer,
                               corpus.experiment_files_voc_explain,
                               loss=loss_fn)

            # network = LSTM('to_test/weights' + str(i) + '.hdf5', False)
            # network.do_explain(optimizer,
            #                      corpus.experiment_files_voc_explain,
            #                      loss=loss_fn)


if __name__ == "__main__":
    main()
