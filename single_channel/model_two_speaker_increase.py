from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, Bidirectional, multiply
from keras.layers.core import Activation, Flatten, Lambda, Dropout
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D, UpSampling2D
from keras.layers.recurrent import Recurrent
from keras.layers.recurrent import LSTM as keras_LSTM
from keras.layers.recurrent import GRU as keras_GRU
from keras.layers.wrappers import TimeDistributed
# from keras.engine import merge
from keras import backend as K

import numpy as np

from network import Network

class Test(Network):
    def __init__(self, network, s1=True):
        super(Test, self).__init__(network)
        self.network = network

    def model(self):
        mix = Input(shape=(None, self.features), name='input')
        model = Model(input=[mix], output=[mix])
        return model

class GRU_net(Network):
    def __init__(self, network, s1=True):
        super(GRU_net, self).__init__(network)
        self.network = network

    def model(self):
        mix = Input(shape=(None, self.features), name='input')
        gru1 = keras_GRU(self.units[0], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(mix)
        gru2 = keras_GRU(self.units[1], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(gru1)
        gru3 = keras_GRU(self.units[2], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(gru2)
        gru4 = keras_GRU(self.units[3], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(gru3)

        speaker_1, speaker_2 = separation_layers(self.features, mix, gru4)
        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model


class LSTM_net(Network):
    """docstring for LSTM"""
    def __init__(self, network, s1=True):
        super(LSTM_net, self).__init__(network)
        self.network = network
        self.single_speaker=s1

    def model(self):
        mix = Input(shape=(None, self.features), name='input')
        lstm1 = keras_LSTM(self.units[0], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT, name='lstm1')(mix)
        lstm2 = keras_LSTM(self.units[1], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT, name='lstm2')(lstm1)
        lstm3 = keras_LSTM(self.units[2], return_sequences=True,
                          dropout=self.DROPOUT,
                          recurrent_dropout=self.RDROPOUT, name='lstm3')(lstm2)
        lstm4 = keras_LSTM(self.units[3], return_sequences=True,
                          dropout=self.DROPOUT,
                          recurrent_dropout=self.RDROPOUT, name='lstm4')(lstm3)

        tdd1 = TimeDistributed(Dense(2*self.features, activation="tanh"), name='dense1')(lstm4)
        tdd2 = TimeDistributed(Dense(2*self.features, activation="tanh"), name='dense2')(tdd1)


        speaker_1, speaker_2 = separation_layers(self.features, mix, tdd2)
        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model

class DNN(Network):
    """docstring for DNN"""
    def __init__(self, network, s1=True):
        super(DNN, self).__init__(network)
        self.network = network
        self.single_speaker=s1

    def model(self):
        mix = Input(shape=(None, self.features), name='input')
        dnn1 = TimeDistributed(Dense(self.units[0], activation='tanh'), name='dense1')(mix)
        dnn2 = TimeDistributed(Dense(self.units[0], activation='tanh'), name='dense2')(dnn1)
        tdd1 = TimeDistributed(Dense(self.units[0], activation="tanh"), name='dense3')(dnn2)
        tdd2 = TimeDistributed(Dense(self.units[0], activation="tanh"), name='dense4')(tdd1)
        tdd2 = TimeDistributed(Dense(2*self.features, activation="tanh"), name='dense5')(tdd2)

        speaker_1, speaker_2 = separation_layers(self.features, mix, tdd2)
        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model

class CNN(Network):
    """docstring for DNN"""
    def __init__(self, network, s1=True):
        super(CNN, self).__init__(network)
        self.network = network
        self.single_speaker=s1

    def model(self):
        kernels = 64
        kernel_width = 3
        kernel_height = 3

        mix = Input(batch_shape=(None, self.sequences, self.timesteps, self.features))

        conv1 = Convolution2D(kernels, kernel_height, kernel_width, border_mode='same')(mix)
        conv2 = Convolution2D(kernels, kernel_height, kernel_width, border_mode='same')(conv1)
        conv3 = Convolution2D(kernels, kernel_height, kernel_width, border_mode='same')(conv2)
        #conv3 = Convolution2D(kernels*2, kernel_height, kernel_width, border_mode='same')(conv3)
        #conv3 = Convolution2D(kernels*2, kernel_height, kernel_width, border_mode='same')(conv3)
        #conv3 = Convolution2D(kernels, kernel_height, kernel_width, border_mode='same')(conv3)
        #conv3 = Convolution2D(kernels, kernel_height, kernel_width, border_mode='same')(conv3)
        #trans = Convolution2D(self.sequences, kernel_height, kernel_width, border_mode='same')(conv3)
        trans = Convolution2D(1, kernel_height, kernel_width, border_mode='same')(conv3)

        dnn1 = TimeDistributed(Dense(500, activation='sigmoid'))(trans)
        #drop1 = Dropout(0.5)(dnn1)
        dnn2 = TimeDistributed(Dense(800, activation='sigmoid'))(dnn1)
        #dnn2 = TimeDistributed(Dense(200, activation='relu'))(drop1)
        dnn2 = TimeDistributed(Dense(200, activation='sigmoid'))(dnn2)

        speaker_1, speaker_2 = separation_layers(self.features, mix, dnn2)
        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model


class RCNN(Network):
    """docstring for DNN"""
    def __init__(self, network, s1=True):
        super(RCNN, self).__init__(network)
        self.network = network
        self.single_speaker=s1

    def model(self):
        kernels = 64
        kernel_width = 2
        kernel_height = 2

        mix = Input(batch_shape=(None, self.sequences, self.timesteps, self.features))
        conv1 = Convolution2D(kernels, kernel_height, kernel_width, border_mode='same')(mix)
        conv2 = Convolution2D(1, kernel_height, kernel_width, border_mode='same')(conv1)
        dnn1 = TimeDistributed(Dense(self.features, activation='relu'))(conv2)
        resh1 = Reshape((self.timesteps, self.features), input_shape=(1,self.sequences, self.timesteps, self.features))(dnn1)
        lstm1 = keras_LSTM(self.features, return_sequences=True)(resh1)
        lstm2 = keras_LSTM(self.features, return_sequences=True, go_backwards=True)(lstm1)
        drop1 = Dropout(0.5)(lstm2)
        # speaker_1, speaker_2 = separation_layers(self.features, mix, lstm2)
        speaker_1, speaker_2 = separation_layers(self.features, mix, drop1)
        model = Model(input=[mix], output=[speaker_1, speaker_2])
        #model = Model(input=[mix], output=[resh1])

        return model

def separation_layers(features, input_layer, bottom_layer):
    increase = TimeDistributed(Dense(2*features, activation="sigmoid"), name='increase_mix')(bottom_layer)
    tdd1 = TimeDistributed(Dense(features, activation="sigmoid"), name='dense_speaker_1')(increase)
    tdd2 = TimeDistributed(Dense(features, activation="sigmoid"), name='dense_speaker_2')(increase)
    speaker_1 = Lambda(function=lambda x: mask(tdd1, tdd2, input_layer), name='speaker_1')(tdd1)
    speaker_2 = Lambda(function=lambda x: mask(tdd2, tdd1, input_layer), name='speaker_2')(tdd2)

    return speaker_1, speaker_2


def mask(predicted_1, predicted_2, mix):
    the_mask = K.pow(K.abs(predicted_1),2) / (K.pow(K.abs(predicted_1),2) + K.pow(K.abs(predicted_2),2))
    # return merge([the_mask,mix[0,0]], mode= "mul")
    return multiply([the_mask,mix])

def ideal_mask(predicted_1, predicted_2, mix):
    mags = np.dstack((predicted_1, predicted_2))
    mask = mags >= np.max(mags, axis=2, keepdims=True)
    return mix * mask[:,:,0]
