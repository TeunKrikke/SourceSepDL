from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, multiply
from keras.layers.core import Activation, Flatten, Lambda, Dropout
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D, UpSampling2D
from keras.layers.recurrent import GRU, Recurrent
from keras.layers.recurrent import LSTM as keras_LSTM
from keras.layers.wrappers import TimeDistributed
# from keras.engine import merge
from keras import backend as K

import numpy as np

from network import Network

class LSTM_single_speaker(Network):
    """docstring for LSTM"""
    def __init__(self, network):
        super(LSTM_single_speaker, self).__init__(network)
        self.network = network

    def model(self):
        mix = Input(batch_shape=(self.sequences, self.timesteps, self.features))
        lstm1 = keras_LSTM(self.features, return_sequences=True)(mix)
        lstm2 = keras_LSTM(self.features, return_sequences=True)(lstm1)
        # drop = Dropout(0.5)(lstm2)

        gt_speaker2 = Input(batch_shape=(self.sequences, self.timesteps, self.features))

        speaker_1 = separation_layers(self.features, mix, lstm2, gt_speaker2)

        model = Model(input=[mix, gt_speaker2], output=[speaker_1])

        return model

class LSTM_backwards_single_speaker(Network):
    """docstring for LSTM_backwards"""
    def __init__(self, network):
        super(LSTM_backwards_single_speaker, self).__init__(network)
        self.network = network

    def model(self):
        mix = Input(batch_shape=(self.sequences, self.timesteps, self.features))
        lstm1 = keras_LSTM(self.features, return_sequences=True)(mix)
        lstm2 = keras_LSTM(self.features, return_sequences=True, go_backwards=True)(lstm1)
        lstm3 = keras_LSTM(self.features, return_sequences=True)(lstm2)
        lstm4 = keras_LSTM(self.features, return_sequences=True, go_backwards=True)(lstm3)
        # drop = Dropout(0.5)(lstm2)

        gt_speaker2 = Input(batch_shape=(self.sequences, self.timesteps, self.features))
        speaker_1 = separation_layers(self.features, mix, lstm4, gt_speaker2)

        model = Model(input=[mix, gt_speaker2], output=[speaker_1])

        return model

class DNN_single_speaker(Network):
    """docstring for DNN"""
    def __init__(self, network):
        super(DNN, self).__init__(network)
        self.network = network

    def model(self):
        mix = Input(batch_shape=(self.sequences, self.timesteps, self.features))
        dnn1 = TimeDistributed(Dense(150, activation='relu'))(mix)
        drop = Dropout(0.5)(dnn1)
        dnn2 = TimeDistributed(Dense(150, activation='relu'))(drop)
        bn = BatchNormalization()(dnn2)

        gt_speaker2 = Input(batch_shape=(self.sequences, self.timesteps, self.features))
        speaker_1 = separation_layers(self.features, mix, bn, gt_speaker2)

        model = Model(input=[mix, gt_speaker2], output=[speaker_1])


        return model

class CNN_single_speaker(Network):
    """docstring for DNN"""
    def __init__(self, network):
        super(CNN, self).__init__(network)
        self.network = network

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

        gt_speaker2 = Input(batch_shape=(self.sequences, self.timesteps, self.features))
        speaker_1 = separation_layers(self.features, mix, dnn2, gt_speaker2)

        model = Model(input=[mix, gt_speaker2], output=[speaker_1])


        return model


class RCNN_single_speaker(Network):
    """docstring for DNN"""
    def __init__(self, network):
        super(RCNN, self).__init__(network)
        self.network = network

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

        gt_speaker2 = Input(batch_shape=(self.sequences, self.timesteps, self.features))

        speaker_1 = separation_layers(self.features, mix, drop1, gt_speaker2)

        model = Model(input=[mix, gt_speaker2], output=[speaker_1])


        return model

def separation_layers(features, input_layer, bottom_layer, ground_truth):
    tdd1 = TimeDistributed(Dense(features, activation="sigmoid"))(bottom_layer)
    speaker_1 = Lambda(function=lambda x: mask(tdd1, ground_truth, input_layer))(tdd1)

    return speaker_1

def mask(predicted_1, predicted_2, mix):
    the_mask = K.pow(K.abs(predicted_1),2) / (K.pow(K.abs(predicted_1),2) + K.pow(K.abs(predicted_2),2))
    # return merge([the_mask,mix[0,0]], mode= "mul")
    return multiply([the_mask,mix[0,0]])

def ideal_mask(predicted_1, predicted_2, mix):
    mags = np.dstack((predicted_1, predicted_2))
    mask = mags >= np.max(mags, axis=2, keepdims=True)
    return mix * mask[:,:,0]
