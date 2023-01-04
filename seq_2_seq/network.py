#CNN for source separation

from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input
from keras.layers.core import Activation, Flatten, Lambda, Dropout
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D, UpSampling2D
from keras.layers.recurrent import GRU, Recurrent
from keras.layers.recurrent import LSTM as keras_LSTM
from keras.layers.wrappers import TimeDistributed
from keras.engine import merge
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import TensorBoard
# from keras.datasets import mnist
import tensorflow as tf

import numpy as np
from PIL import Image
import argparse
import math
from os import listdir
from os.path import isfile, isdir, join
import matplotlib.pyplot as plt
import mir_eval
from scipy.io import wavfile

import matplotlib.pyplot as plt

import glob
import os

#from libroas.core import stft, istft
#from librosa.output import write_wav

from utils import create_mixture, do_STFT_on_data, create_batches, write_wav, istft
import read_ac_export as rae
from multi_gpu import make_parallel

import mir_eval
#"image_dim_ordering": "th", 

class Network(object):
    """Network class which is implemented by the different networks"""
    def __init__(self, network):
        super(Network, self).__init__()
        self.network = network
        self.sequences = 1
        self.timesteps = None
        self.features = 855
        self.win = 255
        self.overlap = 0.5
        self.batch_size = None
        
        self.batch_hop = 500
        
        self.SDR = [0, 0]
        self.SIR = [0, 0]
        self.SAR = [0, 0]
        self.num_mics = 1
        self.mixtures = 100
        
    

    def train(self, optim, corpus, loss='mse', BATCH_SIZE=20, EPOCHS=1001):
        sess = tf.Session()
        K.set_session(sess)
        callback = TensorBoard(log_dir='logs/epoch_'+self.network)
        batch_callback = TensorBoard(log_dir='logs/batch_'+self.network)

        model = make_parallel(self.model(),2)

        model.validation_data= None
        
        model.compile(loss=loss, optimizer=optim)
        noise = np.zeros((BATCH_SIZE, 100))

        BATCH_SIZE_FOR_FEEDING = BATCH_SIZE / 2

        win=255
        overlap=0.5
        combination = 0


        d_losses = []
        g_losses = []

        callback.set_model(model)
        callback.set_params({
            'batch_size': BATCH_SIZE,
            'nb_epoch': EPOCHS,
            'verbose': True,
            'do_validation': False,
            
        })
        callback.on_train_begin()

       

        for epoch in range(EPOCHS+1):
            epoch_logs = {}
            # callback.on_epoch_begin(epoch)
            print "Epoch is", epoch


            loss = 0

            for mixture in range(self.mixtures):
                #need to do zero padding
                batch_logs = {}

                mix, speaker_1_padded, speaker_2_padded = create_mixture(corpus)
                
                mix_batched, speaker_1_batched, speaker_2_batched = create_batches(mix, speaker_1_padded, speaker_2_padded, batch_hop=self.batch_hop)
                
                # print mix_batched.shape
                for batch_mix, batch_s1, batch_s2 in zip(mix_batched, speaker_1_batched, speaker_2_batched): 
                    # print batch_mix.shape
                    batch_mix = batch_mix.reshape(1,1,batch_mix.shape[0])
                    batch_s1 = batch_s1.reshape(1,1,batch_s1.shape[0])
                    batch_s2 = batch_s2.reshape(1,1,batch_s2.shape[0])
                    if not self.single_speaker:
                        loss = model.train_on_batch([batch_mix], [batch_s1, batch_s2])
                    else:
                        loss = model.train_on_batch([batch_mix, batch_s1], [batch_s2])
                print 'mixture', mixture, 'loss', loss

            # epoch_logs['loss_1'] = loss[0]    
            # epoch_logs['loss_2'] = loss[1]
            # epoch_logs['loss_3'] = loss[2]
            # # epoch_logs['loss'] = loss
            for test_index in range(100):
                self.test_network(model, corpus)
            print '!!!!!SDR = {}, SIR = {}, SAR = {}'.format(self.SDR, self.SIR, self.SAR) 

    def test_network(self, model, corpus): 
        mix, speaker_1_padded, speaker_2_padded = create_mixture(corpus)
        
        mix_batched, speaker_1_batched, speaker_2_batched = create_batches(mix, speaker_1_padded, speaker_2_padded)
        index = 0

        for batch_mix, batch_s1, batch_s2 in zip(mix_batched, speaker_1_batched, speaker_2_batched):
            batch_mix = batch_mix.reshape(1, 1, batch_mix.shape[0])
            batch_s1 = batch_s1.reshape(1, 1, batch_s1.shape[0])
            if not self.single_speaker:
                predicted_signal_1,predicted_signal_2 = model.predict([batch_mix]) 
            else:
                predicted_signal_1 = model.predict([batch_mix, batch_s1]) 
            # print predicted_signal_1.shape
            if index == 0: 
                signal_1 = predicted_signal_1[0,0,:self.batch_hop] 
                if not self.single_speaker: 
                    signal_2 = predicted_signal_2[0,0,:self.batch_hop] 
            elif index == mix_batched.shape[0]-1: 
                signal_1 = np.concatenate((signal_1, predicted_signal_1[0,0]))
                if not self.single_speaker: 
                    signal_2 = np.concatenate((signal_2, predicted_signal_2[0,0])) 
            else: 
                signal_1 = np.concatenate((signal_1, predicted_signal_1[0,0,:self.batch_hop])) 
                if not self.single_speaker:
                    signal_2 = np.concatenate((signal_2, predicted_signal_2[0,0,:self.batch_hop])) 
            index += 1
 
        print signal_1.shape 
        print mix.shape 
        # signal_1 = istft(signal_1, windowsize=self.win, overlap=self.overlap) 
        # if not self.single_speaker:
        #     signal_2 = istft(signal_2, windowsize=self.win, overlap=self.overlap) 
 
        speaker_2_padded = speaker_2_padded[:len(signal_1)]
        if not self.single_speaker: 
            speaker_1_padded = speaker_1_padded[:len(signal_1)] 

        write_wav(signal_1, filename=self.network+'_signal1.wav')
        if not self.single_speaker:
            write_wav(signal_2, filename=self.network+'_signal2.wav')
 
        try: 
            if self.single_speaker:
                bss = mir_eval.separation.bss_eval_sources( 
                    speaker_1_padded, signal_1) 
            else:
                bss = mir_eval.separation.bss_eval_sources( 
                    np.stack([speaker_1_padded, speaker_2_padded]), np.stack([signal_1,signal_2])) 
            SDR = bss[0] 
            self.SDR = self.SDR + SDR 
            SIR = bss[1] 
            self.SIR = self.SIR + SIR 
            SAR = bss[2] 
            self.SAR = self.SAR + SAR 
            # print '!!!!!SDR = {}, SIR = {}, SAR = {}, perm = {}'.format(*bss) 
        except ValueError: 
            print "outcome is 0 (zero)" 
            
    


