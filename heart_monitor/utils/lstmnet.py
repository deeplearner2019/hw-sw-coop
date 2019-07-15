from keras.models import Model
from keras.layers import Input, LSTM, CuDNNLSTM, Dense

class LstmNet:
    @staticmethod
    def build(input_shape, n_layers, c_units, n_classes):
        '''
        Build a LSTM model for classification with n_layers of LSTM followed by a final FC layer.

        Parameters
        ----------
        input_shape : tuple, shape of the input data
        n_layers : int, number of hidden LSTM layers
        c_units : list, number of hidden units to consider for each layer
        n_classes : int, number of classes to predict
        '''

        inputs = Input(shape=input_shape, name='input')

        if n_layers == 1:
            layer = CuDNNLSTM(c_units[0], name='lstm')(inputs) 

        if n_layers > 1:
            layer = CuDNNLSTM(c_units[0], name='lstm1', return_sequences=True)(inputs)
            for i in range(n_layers-2):
                layer = CuDNNLSTM(c_units[i+1], return_sequences=True, name='lstm{}'.format(i+2))(layer)
            layer = CuDNNLSTM(c_units[-1], name='lstm{}'.format(n_layers))(layer)

        dense = Dense(n_classes, name='dense', activation='softmax')(layer)

        model = Model(inputs=inputs, outputs=dense)
        return model