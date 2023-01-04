# cost_functions.py
from keras import backend as K
def isd(y_true, y_pred):
    """
     Itakura-Saito divergence
    """
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    # return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    return (y_true/y_pred) - K.log(y_true/y_pred) - 1

def csd(y_true, y_pred):
    """
    complex isotropic Cauchy distance
    """
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)


    return (3/2) * K.log(y_true**2 + y_pred**2) - K.log(y_true)

def wd(y_true, y_pred):
    """
    Wasserstein distance
    """
    # try without clipping
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)


    return K.mean(y_true * y_pred)
