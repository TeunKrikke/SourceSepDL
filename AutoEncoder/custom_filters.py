from keras.layers import Dense, Lambda, multiply
from keras.layers.wrappers import TimeDistributed
from keras import backend as K


def separation_layers(features, input_layer, bottom_layer):
    """
        Separation layer/filter using a wiener filter with 3 sigmoid
        activated dense layers

        Keyword arguments:
        features -- number of FT units
        input_layer -- the original mixture
        bottom_layer -- the last layer of the network

        returns:
        speaker_1 -- prediction of speaker 1 signal
        speaker_2 -- prediction of speaker 2 signal
    """
    increase = TimeDistributed(Dense(2*features, activation="sigmoid"), name='mix_increase')(bottom_layer)
    tdd1 = TimeDistributed(Dense(features, activation="sigmoid"), name='speaker_1_dense')(increase)
    tdd2 = TimeDistributed(Dense(features, activation="sigmoid"), name='speaker_2_dense')(increase)
    speaker_1 = Lambda(function=lambda x: mask(tdd1, tdd2, input_layer),
                       name='speaker_1')(tdd1)
    speaker_2 = Lambda(function=lambda x: mask(tdd2, tdd1, input_layer),
                       name='speaker_2')(tdd2)

    return speaker_1, speaker_2


def separation_layers_no(features, input_layer, bottom_layer):
    """
        Separation layer/filter no filter but 5 sigmoid
        activated dense layers

        Keyword arguments:
        features -- number of FT units
        input_layer -- the original mixture
        bottom_layer -- the last layer of the network

        returns:
        speaker_1 -- prediction of speaker 1 signal
        speaker_2 -- prediction of speaker 2 signal
    """
    increase = TimeDistributed(Dense(2*features, activation="sigmoid"), name='mix_increase')(bottom_layer)
    tdd1 = TimeDistributed(Dense(200, activation="sigmoid"), name='speaker_1_dense')(increase)
    tdd2 = TimeDistributed(Dense(200, activation="sigmoid"), name='speaker_2_dense')(increase)
    speaker_1 = TimeDistributed(Dense(features, activation="sigmoid"),
                                name='speaker_1')(tdd1)
    speaker_2 = TimeDistributed(Dense(features, activation="sigmoid"),
                                name='speaker_2')(tdd2)

    return speaker_1, speaker_2


def separation_layers_ideal(features, input_layer, bottom_layer):
    """
        Separation layer/filter using an ideal filter with 3 sigmoid
        activated dense layers

        Keyword arguments:
        features -- number of FT units
        input_layer -- the original mixture
        bottom_layer -- the last layer of the network

        returns:
        speaker_1 -- prediction of speaker 1 signal
        speaker_2 -- prediction of speaker 2 signal
    """
    increase = TimeDistributed(Dense(2*features, activation="sigmoid"), name='mix_increase')(bottom_layer)
    tdd1 = TimeDistributed(Dense(features, activation="sigmoid"), name='speaker_1_dense')(increase)
    tdd2 = TimeDistributed(Dense(features, activation="sigmoid"), name='speaker_2_dense')(increase)
    speaker_1 = Lambda(function=lambda x: ideal_mask(tdd1, tdd2, input_layer),
                       name='speaker_1')(tdd1)
    speaker_2 = Lambda(function=lambda x: ideal_mask(tdd2, tdd1, input_layer),
                       name='speaker_2')(tdd2)

    return speaker_1, speaker_2


def separation_layers_tanh(features, input_layer, bottom_layer):
    """
        Separation layer/filter using a wiener filter with 3 layers of tanh
        activated dense layers

        Keyword arguments:
        features -- number of FT units
        input_layer -- the original mixture
        bottom_layer -- the last layer of the network

        returns:
        speaker_1 -- prediction of speaker 1 signal
        speaker_2 -- prediction of speaker 2 signal
    """
    increase = TimeDistributed(Dense(2*features, activation="tanh"), name='mix_increase')(bottom_layer)
    tdd1 = TimeDistributed(Dense(features, activation="tanh"), name='speaker_1_dense')(increase)
    tdd2 = TimeDistributed(Dense(features, activation="tanh"), name='speaker_2_dense')(increase)
    speaker_1 = Lambda(function=lambda x: mask(tdd1, tdd2, input_layer),
                       name='speaker_1')(tdd1)
    speaker_2 = Lambda(function=lambda x: mask(tdd2, tdd1, input_layer),
                       name='speaker_2')(tdd2)

    return speaker_1, speaker_2


def separation_layers_linear(features, input_layer, bottom_layer):
    """
        Separation layer/filter using a wiener filter with 3 linear
        activated dense layers

        Keyword arguments:
        features -- number of FT units
        input_layer -- the original mixture
        bottom_layer -- the last layer of the network

        returns:
        speaker_1 -- prediction of speaker 1 signal
        speaker_2 -- prediction of speaker 2 signal
    """
    increase = TimeDistributed(Dense(2*features, activation="linear"), name='mix_increase')(bottom_layer)
    tdd1 = TimeDistributed(Dense(features, activation="linear"), name='speaker_1_dense')(increase)
    tdd2 = TimeDistributed(Dense(features, activation="linear"), name='speaker_2_dense')(increase)
    speaker_1 = Lambda(function=lambda x: mask(tdd1, tdd2, input_layer),
                       name='speaker_1')(tdd1)
    speaker_2 = Lambda(function=lambda x: mask(tdd2, tdd1, input_layer),
                       name='speaker_2')(tdd2)

    return speaker_1, speaker_2


def mask(predicted_1, predicted_2, mix):
    """
        Masking using a wiener filter

        Keyword arguments:
        predicted_1 -- filter prediction of speaker 1 as learned by the network
        predicted_2 -- filter prediction of speaker 2 as learned by the network
        mix -- the original mixture

        returns:
        signal of predicted 1
    """
    the_mask = K.pow(K.abs(predicted_1), 2) / (K.pow(K.abs(predicted_1), 2) +
                                               K.pow(K.abs(predicted_2), 2))
    # return merge([the_mask,mix[0,0]], mode= "mul")
    return multiply([the_mask, mix])


def ideal_mask(predicted_1, predicted_2, mix):
    """
        Masking using a ideal filter

        Keyword arguments:
        predicted_1 -- filter prediction of speaker 1 as learned by the network
        predicted_2 -- filter prediction of speaker 2 as learned by the network
        mix -- the original mixture

        returns:
        signal of predicted 1
    """
    # mags = np.dstack((predicted_1, predicted_2))
    # mask = mags >= np.max(mags, axis=2, keepdims=True)

    # return mix * mask[:,:,0]

    return multiply([predicted_1, mix])
