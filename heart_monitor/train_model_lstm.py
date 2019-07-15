from utils import config
from utils.lstmnet import LstmNet
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
from keras.optimizers import Adam


N_TIMESTEPS = 64
N_FEATURES = 64


def load_data_train():
    if os.path.isfile(os.path.join(config.BASE_PATH, 'data_heart.npz')) == True:
        data = np.load(os.path.join(config.BASE_PATH, 'data_heart.npz'))
        X_train = data['X_train']
        Y_train = data['Y_train']
    return X_train, Y_train

def load_data_val():
    if os.path.isfile(os.path.join(config.BASE_PATH, 'data_heart.npz')) == True:
        data = np.load(os.path.join(config.BASE_PATH, 'data_heart.npz'))
        X_val = data['X_val']
        Y_val = data['Y_val']
    return X_val, Y_val
                      
def load_data_test():
    if os.path.isfile(os.path.join(config.BASE_PATH, 'data_heart.npz')) == True:
        data = np.load(os.path.join(config.BASE_PATH, 'data_heart.npz'))
        X_test = data['X_test']
        Y_test = data['Y_test']
    return X_test, Y_test



def main(X_train, Y_train, X_val, Y_val, X_test, Y_test, n_layers, h_units):
    '''
    Train and predict.
    
    Parameters
    ----------
    n_layers : int, number of hidden LSTM layers
    c_units : int, number of hidden units for each layer
    '''
    
    batch_size = 64
    epochs = 100
    
    print('\nModel with {} layer(s) and {} hidden units each'.format(n_layers, h_units))
    start = time()
    model = LstmNet.build(input_shape=(N_TIMESTEPS, N_FEATURES),
                          n_layers = n_layers,
                          c_units = [h_units]*n_layers,
                          n_classes = 2)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.fit(X_train, Y_train,
              validation_data=(X_val, Y_val),
              batch_size=batch_size,
              epochs=epochs,
              verbose=0)
    end = time()
    
    tr_loss = model.history.history['loss'][-1]
    tr_val_loss = model.history.history['val_loss'][-1]
    tr_acc = model.history.history['acc'][-1]
    tr_val_acc = model.history.history['val_acc'][-1]
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
    X_val, Y_val = load_data_val()
    X_test, Y_test = load_data_test()
    
    main(X_train, Y_train, X_val, Y_val, X_test, Y_test, args.layers, args.units)