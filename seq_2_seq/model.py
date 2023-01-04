from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input, RepeatVector, recurrent, concatenate, add, multiply

from keras.engine.topology import Layer

from keras import backend as K

from network import Network

from utils import util_slice, get_slices, to_list



class Basic_Seq2Seq(Network):
    """docstring for Basic_Seq2Seq"""
    def __init__(self, output_dim, output_length, hidden_dim=None, input_shape=None,
                 depth=1, dropout=0.0, unit=recurrent.LSTM):
        super(Basic_Seq2Seq, self).__init__(network)
        self.arg = arg

    def model(self):
        model = Sequential()
        #encoder bit
        model.add(unit(self.hidden_units, input_shape=(self.input_shape[0],self.input_shape[-1])))
        for _ in range(1, self.depth):
            model.add(Dropout(self.dropout))
            model.add(unit(self.hidden_units))

        #decoder bit
        model.add(Dropout(self.dropout, batch_input_shape=(self.input_shape[0], self.hidden_units)))        
        for _ in range(1, self.depth):
            model.add(unit(self.hidden_units))
            model.add(Dropout(self.dropout))        
        model.add(unit(output_dim))

        return model   

class Attention_Seq2Seq(Network):
     """docstring for Attention_Seq2Seq"""
    def __init__(self, output_dim, output_length, hidden_dim=None, input_shape=None,
                 depth=1, dropout=0.0, unit=recurrent.LSTM):
        super(Attention_Seq2Seq, self).__init__(network)
        self.arg = arg
          
    def model(self):
        model = Sequential()
        model.add(unit(self.hidden_units, input_shape=(self.input_shape[0],self.input_shape[-1])))
        for _ in range(1, self.depth):
            model.add(Dropout(self.dropout))
            model.add(unit(self.hidden_units))

        #decoder bit
        model.add(Dropout(self.dropout, batch_input_shape=(self.input_shape[0], self.hidden_units)))
        model.add(AttentionCell(output_dim=output_dim, hidden_dim=hidden_dim))        
        for _ in range(1, self.depth):
            model.add(unit(self.hidden_units))
            model.add(Dropout(self.dropout))        
        model.add(unit(output_dim))

        return model

class AttentionCell(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(AttentionCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        input_length = input_shape[1]

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))

        W1 = Dense(hidden_dim * 4)
        W2 = Dense(output_dim)
        W3 = Dense(1)
        U = Dense(hidden_dim * 4)

        C = Lambda(lambda x : K.repeat(x, input_length), output_shape=(input_length, input_dim))(c_tm1)
        _xC = concatenate([x, C])
        _xC = Lambda(lambda x : K.reshape(x, (-1, input_dim+hidden_dim)), output_shape=(input_dim + hidden_dim,))(_xC)

        alpha = W3(_xC)
        alpha = Lambda(lambda x : K.reshape(x, (-1, input_length)), output_shape=(input_length,))(alpha)
        alpha = Activation('softmax')(alpha)

        _x = Lambda(lambda x : K.batch_dot(x[0], x[1], axis(1, 1)), output_shape=(input_dim,))([alpha, x])

        z = add([W1(_x), U(h_tm1)])

        z0, z1, z2, z3 = self.get_slices(z, 4)

        i = Activation('hard_sigmoid')(z0)
        f = Activation('hard_sigmoid')(z0)

        c = add([multiply([f, c_tm1]), multiply([i, Activation('tanh')(z2)])])
        o = Activation('hard_sigmoid')(z3)
        h = multiply([o, Activation('tanh')(c)])
        y = Activation('tanh')(W2(h))

        if type(input_shape) is list:
            self.input_spec = [InputSpec(shape=shape) for shape in input_shape]
            self.model = Model([x, h_tm1, c_tm1], [y, h, c])
        else:
            self.model = Model([x, h_tm1, c_tm1], [y, h, c])
            self.input_spec = [InputSpec(shape=shape) for shape in to_list(input_shape)]

    def call(self, inputs, learning=None):
        return self.model.call(inputs)

    def compute_output_shape(self, input_shape):
        model_inputs = self.model.input
        if type(model_inputs) is list and type(input_shape) is not list:
            input_shape = [input_shape] + list(map(K.int_shape, self.model.input[1:]))
        return self.model.compute_output_shape(input_shape)