from model import Basic_Seq2Seq

from keras.optimizers import SGD, Adam, Adamax, Nadam

from cost_functions import isd, csd

import corpus


def main():

    run_batch_optimizer(loss='mse')
    run_batch_optimizer(loss='kl') 
    # run_batch_optimizer(loss=isd,s1=s1)
    # run_batch_optimizer(loss=csd,s1=s1) 


def run_batch_optimizer(loss='mse'):
    optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    run_batch_networks(optimizer, optim_name='SGD', loss=loss, s1=s1)

    optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    run_batch_networks(optimizer, optim_name='nadam', loss=loss, s1=s1)

    optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    run_batch_networks(optimizer, optim_name='Adamax', loss=loss, s1=s1)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    run_batch_networks(optimizer, optim_name='Adam', loss=loss, s1=s1)

def run_batch_networks(optimizer, optim_name='SGD', loss='mse'):
    
        network = Basic_Seq2Seq('voc_BSeq2Seq_'+optim_name+'_'+str(loss))
        network.train(optimizer, corpus.experiment_files_voc, EPOCHS=100)

if __name__ == "__main__":
    main()
    #main(False)
