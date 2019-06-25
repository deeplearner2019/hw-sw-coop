from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

from os import environ
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
environ["CUDA_VISIBLE_DEVICES"] = '0'
    
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from os.path import join
from pickle import dump, load
from time import time

import numpy as np

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, CuDNNLSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from preprocess import *

# def set_callbacks(save_path, min_delta, patience):
#     """
#     Set callbacks to monitor the model performance
    
#     Parameters
#     ----------
#     save_path : path where the model will be saved
#     min_delta : minimal accepted variation of validation loss to pursue training
#     patience : number of additional epochs to look at before early stopping
#     """
   
#     checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
# #     tbCallBack = TensorBoard(log_dir='log/', histogram_freq=5, write_grads=True)
#     es = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=0)
#     return checkpoint, es

    
def build_lstm(input_shape, n_hidden1, n_hidden2, n_class, activation, optimizer, loss, metrics):
    inputs = Input(input_shape, name='Input')
    lstm1 = LSTM(n_hidden1, return_sequences=True, activation=activation, name='Lstm1')(inputs)
    lstm2 = LSTM(n_hidden2, return_sequences=False, activation=activation, name='Lstm2')(lstm1)
    dense = Dense(n_class, activation='softmax', name='Dense')(lstm2)
    model = Model(inputs=inputs, outputs=dense)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model


def build_cudnnlstm(input_shape, n_hidden1, n_hidden2, n_class, optimizer, loss, metrics):
    inputs = Input(input_shape, name='Input')
    lstm1 = CuDNNLSTM(n_hidden1, return_sequences=True, name='cudnnLstm1')(inputs)
    lstm2 = CuDNNLSTM(n_hidden2, return_sequences=False, name='cudnnLstm2')(lstm1)
    dense = Dense(n_class, activation='softmax', name='Dense')(lstm2)
    model = Model(inputs=inputs, outputs=dense)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model


def plot_loss(model, interval):
    """
    Plot training and validation loss.
    
    Parameters
    ----------
    interval : tuple indicating the start epoch and end epoch to plot
    """
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,5))
    plt.plot(np.arange(interval), model.history.history['loss'][interval[0]:interval[-1]], color='blue')
    plt.plot(np.arange(interval), model.history.history['val_loss'][interval[0]:interval[-1]], color='red')
    plt.show()
    

def main():
    data, labels = buildDataset(5, 100, 3)
    train, test, train_label, test_label = splitDataset(data, labels)
    train, test = MinMaxSc(train, test)
    # Scale between [-1,1]
    train = 2*train-1
    test = 2*test-1
    print('Training set : {} - Test set : {}'.format(train.shape, test.shape))
    
    model = build_cudnnlstm(train.shape[-2:], n_hidden1=100, n_hidden2=100, n_class=14,
                       optimizer=Adam(), loss='categorical_crossentropy', metrics='categorical_accuracy') 

    model.fit(train, train_label, batch_size=64, epochs=10, validation_split=0.2,
              verbose=1)
    model.save('lstm_classifier.h5')
    
    print('\nPREDICTION')
    model.evaluate(test, test_label, verbose=1)
    y = model.predict(test)
    
    return model, y


if __name__=='___main__':
    main()
