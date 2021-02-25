from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, Bidirectional, multiply
from keras.layers import Conv2D
from keras.layers.core import Activation, Flatten, Lambda, Dropout
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU as keras_GRU
from keras.layers.recurrent import LSTM as keras_LSTM
from keras.layers.wrappers import TimeDistributed
# from keras.engine import merge
from keras import backend as K

import numpy as np

from network import Network

class GRU_net(Network):
    """Gated recurrent unit network. This inherets from network"""
    def __init__(self, network, filter):
        """
            The bare minimum for initializing this network.

            Keyword arguments:
            network -- the name of the network so we can save the weights.
            filter -- the name of the filter that is being used byt the network
            to do source separation.
        """
        super(GRU_net, self).__init__(network)
        self.network = network
        self.filter = filter

    def model(self):
        """
            The definition of the network. We are using the Fuctional API from
            keras. It inherets most of the things from Network where the
            number of units per layer are defined and the dropout parameters.

            Returns:
            model -- the network architecture.
        """
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

        speaker_1, speaker_2 = self.filter(self.features, mix, gru4)
        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model


class LSTM_net(Network):
    """LSTM network using standard LSTM units"""
    def __init__(self, network, filter):
        """
            The bare minimum for initializing this network.

            Keyword arguments:
            network -- the name of the network so we can save the weights.
            filter -- the name of the filter that is being used byt the network
            to do source separation.
        """
        super(LSTM_net, self).__init__(network)
        self.network = network
        self.filter = filter


    def model(self):
        """
            The definition of the network. We are using the Fuctional API from
            keras. It inherets most of the things from Network where the
            number of units per layer are defined and the dropout parameters.

            Returns:
            model -- the network architecture.
        """
        mix = Input(shape=(None, self.features), name='input')
        lstm1 = keras_LSTM(self.units[0], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(mix)
        lstm2 = keras_LSTM(self.units[1], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(lstm1)
        lstm3 = keras_LSTM(self.units[2], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(lstm2)
        lstm4 = keras_LSTM(self.units[3], return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(lstm3)
        tdd1 = TimeDistributed(Dense(self.features, activation='tanh'), name="dense1")(lstm4)
        tdd2 = TimeDistributed(Dense(self.features, activation='tanh'), name="dense2")(tdd1)

        speaker_1, speaker_2 = self.filter(self.features, mix, tdd2)
        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model


class LSTM_pobs_net(Network):
    """LSTM using the probabilies of the mixture and tries to learn a clean
        separation based on that. It takes a second input which is the original
        mixture and makes the network look more like a NMF.
    """
    def __init__(self, network, filter):
        """
            The bare minimum for initializing this network.

            Keyword arguments:
            network -- the name of the network so we can save the weights.
            filter -- the name of the filter that is being used byt the network
            to do source separation.
        """
        super(LSTM_pobs, self).__init__(network)
        self.network = network
        self.filter = filter

    def model(self):
        """
            The definition of the network. We are using the Fuctional API from
            keras. It inherets most of the things from Network where the
            number of units per layer are defined and the dropout parameters.

            Returns:
            model -- the network architecture.
        """
        pobs = Input(shape=(None, self.features), name='input')
        mix = Input(shape=(None, self.features), name='true_input')
        lstm1 = keras_LSTM(self.units, return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(pobs)
        lstm2 = keras_LSTM(self.units, return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(lstm1)
        lstm3 = keras_LSTM(self.units, return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(lstm2)
        lstm4 = keras_LSTM(self.units, return_sequences=True,
                           dropout=self.DROPOUT,
                           recurrent_dropout=self.RDROPOUT)(lstm3)
        #  drop = Dropout(0.5)(lstm2)

        speaker_1, speaker_2 = self.filter(self.features, mix, lstm4)
        model = Model(input=[pobs, mix], output=[speaker_1, speaker_2])

        return model


class LSTM_backwards_net(Network):
    """LSTM network with forward and backward connections"""
    def __init__(self, network, filter):
        """
            The bare minimum for initializing this network.

            Keyword arguments:
            network -- the name of the network so we can save the weights.
            filter -- the name of the filter that is being used byt the network
            to do source separation.
        """
        super(LSTM_backwards, self).__init__(network)
        self.network = network
        self.filter = filter

    def model(self):
        """
            The definition of the network. We are using the Fuctional API from
            keras. It inherets most of the things from Network where the
            number of units per layer are defined and the dropout parameters.

            Returns:
            model -- the network architecture.
        """
        mix = Input(shape=(None, self.features), name='input')
        lstm1 = Bidirectional(keras_LSTM(self.units, return_sequences=True,
                              dropout=self.DROPOUT,
                              recurrent_dropout=self.RDROPOUT))(mix)
        lstm2 = Bidirectional(keras_LSTM(self.units, return_sequences=True,
                              dropout=self.DROPOUT,
                              recurrent_dropout=self.RDROPOUT))(lstm1)
        lstm3 = Bidirectional(keras_LSTM(self.units, return_sequences=True,
                              dropout=self.DROPOUT,
                              recurrent_dropout=self.RDROPOUT))(lstm2)
        lstm4 = Bidirectional(keras_LSTM(self.units, return_sequences=True,
                              dropout=self.DROPOUT,
                              recurrent_dropout=self.RDROPOUT))(lstm3)
        # drop = Dropout(0.5)(lstm2)

        speaker_1, speaker_2 = self.filter(self.features, mix, lstm4)
        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model

class DNN_net(Network):
    """Deep neural network which uses dense layers for the separation"""
    def __init__(self, network, filter):
        """
            The bare minimum for initializing this network.

            Keyword arguments:
            network -- the name of the network so we can save the weights.
            filter -- the name of the filter that is being used byt the network
            to do source separation.
        """
        super(DNN_net, self).__init__(network)
        self.network = network
        self.filter = filter

    def model(self):
        """
            The definition of the network. We are using the Fuctional API from
            keras. It inherets most of the things from Network where the
            number of units per layer are defined and the dropout parameters.

            Returns:
            model -- the network architecture.
        """
        mix = Input(shape=(None, self.features), name='input')
        dnn1 = TimeDistributed(Dense(500, activation='tanh'), name="dense1")(mix)
        dnn2 = TimeDistributed(Dense(300, activation='tanh'), name="dense2")(dnn1)
        dnn3 = TimeDistributed(Dense(self.features, activation='tanh'), name="dense3")(dnn2)
        dnn4 = TimeDistributed(Dense(self.features, activation='tanh'), name="dense4")(dnn3)

        speaker_1, speaker_2 = self.filter(self.features, mix, dnn4)
        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model

class CNN_net(Network):
    """Convolutional neural network using convolutions to do the job"""
    def __init__(self, network, filter):
        """
            The bare minimum for initializing this network.

            Keyword arguments:
            network -- the name of the network so we can save the weights.
            filter -- the name of the filter that is being used byt the nestwork
            to do source separation.
        """
        super(CNN_net, self).__init__(network)
        self.network = network
        self.filter= filter

    def model(self):
        """
            The definition of the network. We are using the Fuctional API from
            keras. It inherets most of the things from Network where the
            number of units per layer are defined and the dropout parameters.

            Returns:
            model -- the network architecture.
        """
        kernels = 64
        kernel_width = 3
        kernel_height = 3

        mix = Input(batch_shape=(None,1,self.timesteps, self.features), name='input')

        conv1 = Conv2D(kernels, (kernel_height, kernel_width), padding='same', data_format='channels_first')(mix)
        conv2 = Conv2D(kernels, (kernel_height, kernel_width), padding='same', data_format='channels_first')(conv1)
        conv3 = Conv2D(kernels, (kernel_height, kernel_width), padding='same', data_format='channels_first')(conv2)
        trans = Conv2D(1, (kernel_height, kernel_width), padding='same', data_format='channels_first')(conv3)
        resh1 = Reshape((self.timesteps, self.features))(trans)
        dnn1 = TimeDistributed(Dense(500, activation='sigmoid'))(trans)
        dnn2 = TimeDistributed(Dense(800, activation='sigmoid'))(dnn1)
        dnn2 = TimeDistributed(Dense(200, activation='sigmoid'))(dnn2)

        speaker_1, speaker_2 = self.filter(self.features, mix, dnn2)

        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model


class RCNN_net(Network):
    """recurrent CNN using both convolutions and lstms"""
    def __init__(self, network, filter):
        """
            The bare minimum for initializing this network.

            Keyword arguments:
            network -- the name of the network so we can save the weights.
            filter -- the name of the filter that is being used byt the network
            to do source separation.
        """
        super(RCNN, self).__init__(network)
        self.network = network
        self.filter = filter

    def model(self):
        """
            The definition of the network. We are using the Fuctional API from
            keras. It inherets most of the things from Network where the
            number of units per layer are defined and the dropout parameters.

            Returns:
            model -- the network architecture.
        """
        kernels = 64
        kernel_width = 2
        kernel_height = 2

        mix = Input(shape=(self.timesteps, self.features, self.sequences), name='input')
        conv1 = Convolution2D(kernels, kernel_height, kernel_width, border_mode='same')(mix)
        conv2 = Convolution2D(kernels, kernel_height, kernel_width, border_mode='same')(conv1)
        dnn1 = TimeDistributed(Dense(self.features, activation='relu'))(conv2)
        resh1 = Reshape((self.timesteps, self.features), input_shape=(1,self.sequences, self.timesteps, self.features))(dnn1)
        lstm1 = Bidirectional(keras_LSTM(self.features, return_sequences=True))(resh1)
        lstm2 = Bidirectional(keras_LSTM(self.features, return_sequences=True))(lstm1)
        speaker_1, speaker_2 = self.filter(self.features, mix, lstm2)

        model = Model(input=[mix], output=[speaker_1, speaker_2])

        return model
