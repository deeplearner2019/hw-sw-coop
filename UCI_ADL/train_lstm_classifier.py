import numpy as np
import tensorflow as tf
import pickle
from time import time
import argparse

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras.backend.tensorflow_backend as K
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNLSTM, Dense
from keras.optimizers import Adam

from preprocess_classifier import *


N_TIMESTEPS = 100
N_FEATURES = 3


def load_data_train():
    if os.path.isfile('data/data_adl.npz') == True:
        data = np.load('data/data_adl.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
    return X_train, Y_train

def load_data_test():
    if os.path.isfile('data/data_adl.npz') == True:
        data = np.load('data/data_adl.npz')
        X_test = data['X_test']
        Y_test = data['Y_test']
    return X_test, Y_test


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


def main(X_train, Y_train, X_test, Y_test, n_layers, h_units):
    '''
    Train and predict.
    
    Parameters
    ----------
    n_layers : int, number of hidden LSTM layers
    c_units : int, number of hidden units for each layer
    '''
    
    batch_size = 128
    epochs = 50
    
    print('\nModel with {} layer(s) and {} hidden units each'.format(n_layers, h_units))
    start = time()
    model = LstmNet.build(input_shape=(N_TIMESTEPS, N_FEATURES),
                          n_layers = n_layers,
                          c_units = [h_units]*n_layers,
                          n_classes = 14)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train,
              validation_split=0.2,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0)
    end = time()
    
    tr_loss = model.history.history['loss'][-1]
    tr_val_loss = model.history.history['val_loss'][-1]
    tr_acc = model.history.history['categorical_accuracy'][-1]
    tr_val_acc = model.history.history['val_categorical_accuracy'][-1]
    print('Training completed in {}s : loss {}, val_loss {}, acc {}, val_acc {}'
          .format(end-start, tr_loss, tr_val_loss, tr_acc, tr_val_acc))
          
    n_param = model.count_params()
    acc = model.evaluate(X_test, Y_test, verbose=0)[-1]
    print('Model parameters : {}, Prediction accuracy : {}'.format(n_param, acc))
#     model.save('model/heart_lstm_layers{}_units{}_bs{}_epochs{}.h5'
#                .format(n_layers, h_units, batch_size, epochs))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, help='number of LSTM layers')
    parser.add_argument('--units', type=int, help='number of hidden units per layer')
    args = parser.parse_args()
    
    np.random.seed(1)
    tf.set_random_seed(1)
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    K.set_session(sess)
    
    X_train, Y_train = load_data_train()
    X_test, Y_test = load_data_test()
    
    main(X_train, Y_train, X_test, Y_test, args.layers, args.units)