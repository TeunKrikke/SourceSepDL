# cost_functions.py
import tensorflow as tf

_EPSILON = 1e-3

def isd(y_true, y_pred, J, F, N):
    y_true = tf.clip_by_value(y_true, _EPSILON, 1)
    y_pred = tf.clip_by_value(y_pred, _EPSILON, 1)
    
    '''
     Itakura-Saito
     x/y - log(x/y) - 1
    '''
    return (1/(J*F*N)) * tf.reduce_sum(((tf.abs(y_true) ** 2)/y_pred) - tf.log((tf.abs(y_true) ** 2)/y_pred) - 1)
    

def csd(y_true, y_pred, J, F, N):
    y_true = tf.clip_by_value(y_true, _EPSILON, 1)
    y_pred = tf.clip_by_value(y_pred, _EPSILON, 1)
    
    '''
    Cauchy-Schwarz
    3/2 log(x/y) - log(y) 
    '''
    return (1/(J*F*N)) * tf.reduce_sum((3/2) * tf.log(tf.abs(y_true) ** 2 + y_pred) - tf.log(tf.sqrt(y_pred)))

def wd(y_true, y_pred, J, F, N):
    y_true = tf.clip_by_value(y_true, _EPSILON, 1)
    y_pred = tf.clip_by_value(y_pred, _EPSILON, 1)
    
    '''
    Wasserstein
    '''
    return (1/(J*F*N)) * tf.reduce_sum((tf.abs(y_true) ** 2) * y_pred)

def kld(y_true, y_pred, J, F, N):
    y_true = tf.clip_by_value(y_true, _EPSILON, 1)
    y_pred = tf.clip_by_value(y_pred, _EPSILON, 1)
    
    '''
     Kullback-Leibler
     x log(x/y) - x + y
    '''
    return (1/(J*F*N)) * tf.reduce_sum(tf.abs(y_true) * tf.log(tf.abs(y_true)/tf.sqrt(y_pred)) - tf.abs(y_true) + tf.sqrt(y_pred))

def ps(y_true, y_pred, bf_data, J, F, N):
    y_true = tf.clip_by_value(y_true, _EPSILON, 1)
    y_pred = tf.clip_by_value(y_pred, _EPSILON, 1)
    sum_pred = tf.reduce_sum(y_pred, axis=0)
    m = y_pred / sum_pred
    '''
     Phase-Sensitive
    '''
    return (1/(2*J*F*N)) * tf.reduce_sum(tf.abs(m * bf_data - y_true) ** 2)

def mse(y_true, y_pred, J, F, N):
    y_true = tf.clip_by_value(y_true, _EPSILON, 1)
    y_pred = tf.clip_by_value(y_pred, _EPSILON, 1)
    '''
     Mean-squared-error
    '''
    return (1/(2*J*F*N)) * tf.reduce_sum((tf.abs(y_true) - tf.sqrt(y_pred))**2)