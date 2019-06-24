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
from keras.layers import Input, LSTM, CuDNNLSTM, Dense, Masking, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from preprocess_AD import *
import anomaly_detector as ad

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

    
def build_lstm(input_shape, n_hidden1, n_hidden2, activation, optimizer, loss='mse'):
    inputs = Input(input_shape, name='Input')
    mask = Masking(0.)(inputs)
    lstm1 = LSTM(n_hidden1, return_sequences=True, activation=activation, name='Lstm1')(mask)
    lstm2 = LSTM(n_hidden2, return_sequences=False, activation=activation, name='Lstm2')(lstm1)
    model = Model(inputs=inputs, outputs=lstm2)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def build_cudnnlstm(input_shape, n_hidden1, n_hidden2, optimizer, loss='mse'):
    """
    CuDNN implementation for LSTM. Activation is set to default tanh. For other activation functions, use build_lstm.
    """
    
    inputs = Input(input_shape, name='Input')
    lstm1 = CuDNNLSTM(n_hidden1, return_sequences=True, name='cudnnLstm1')(inputs)
    lstm2 = CuDNNLSTM(n_hidden2, return_sequences=False, name='cudnnLstm2')(lstm1)
    model = Model(inputs=inputs, outputs=lstm2)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def train_batch1(model, train, pred_len, epochs, val_split):
    """
    Train model for batch size 1.
    No padding is applied on timeseries
    """
    
    loss = []
    val_loss = []

    minmax = getMinMax(train)

    for e in range(epochs):
        print('\nEpoch {}/{}'.format(e+1, epochs))

        start = time()
        l = 0
        vl = 0

        s_idx = np.arange(len(train))
        np.random.shuffle(s_idx)
        val_raw, tr_raw = train[s_idx][:int(val_split*len(train))], train[s_idx][int(val_split*len(train)):]

        n_samp = len(tr_raw)
        for i in range(len(tr_raw)):
            timeseries = tr_raw[i]
            scaled = MinMaxSc_batch1(timeseries, minmax)
            sTrain, sTarget = scaled[:-pred_len,:][np.newaxis,:,:], scaled[-pred_len:,:].reshape(-1)[np.newaxis,:]
            l += model.train_on_batch(sTrain, sTarget)/n_samp
        print('Time on training set : {}s'.format(time()-start))

        n_val = len(val_raw)
        for timeseries in val_raw:
            scaled = MinMaxSc_batch1(timeseries, minmax)
            sTrain, sTarget = scaled[:-pred_len,:][np.newaxis,:,:], scaled[-pred_len:,:].reshape(-1)[np.newaxis,:]
            vl += model.train_on_batch(sTrain, sTarget)/n_val

        loss.append(l)
        val_loss.append(vl)
        end = time()

    #     if e-earlystop > 0:
    #         avg_loss = np.min(val_loss[-earlystop:])
    #         if vl > avg_loss:
    #             print('\nEarly stopping')
    #             break

        print('Total time : {}s ---------- Loss : {} - Val_loss : {}'.format(end-start, l, vl))

    print('\nTRAINING COMPLETED !')
    
    return model, loss, val_loss


def predict_batch1(model, train, test, pred_len):
    """
    Model prediction for batch size 1.
    No padding is applied on timeseries
    """
    
    evloss = 0
    n_eval = len(test)
    minmax = getMinMax(train)

    y = []
    testTarget = []

    start = time()
    for timeseries in test:
        scaled = MinMaxSc_batch1(timeseries, minmax)
        sTest, sTarget = scaled[:-pred_len,:][np.newaxis,:,:], scaled[-pred_len:,:].reshape(-1)[np.newaxis,:]
        evloss += model.test_on_batch(sTest, sTarget)/n_eval
        y.append(model.predict_on_batch(sTest))
        testTarget.append(sTarget)
    end = time()

    print('Prediction completed in {}s'.format(end-start))
    print("Evaluation loss : {}".format(evloss))
    
    return y, testTarget, evloss


def plot_loss(model, i_min, i_max):
    """
    Plot training and validation loss.
    """
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,5))
    plt.plot(np.arange(i_min, i_max), model.history.history['loss'][i_min:i_max], color='blue')
    plt.plot(np.arange(i_min, i_max), model.history.history['val_loss'][i_min:i_max], color='red')
    plt.show()
    

def main(outlier):
    data, contaminedData, clabels = buildDataset(outlier=outlier, contamination=0.4)
    trainData, trainTarget, testData, testTarget = prepareTS(train=data, test=contaminedData, pred_len=10)
    
    #### Scale between [-1,1] for tanh activation
    trainData = 2*trainData-1
    testData = 2*testData-1
    trainTarget = 2*trainTarget-1
    testTarget = 2*testTarget-1
    ### Supress above block if activation is sigmoid
    
    model = build_cudnnlstm(trainData.shape[-2:], n_hidden1=200, n_hidden2=trainData.shape[-1]*10,
                            optimizer=Adam(), loss='mse') 

    model.fit(trainData, trainTarget, batch_size=64, epochs=150, validation_split=0.2,
              verbose=1)
    model.save('cudnnlstm_ad_{}.h5'.format(outlier))
    
    print('\nPREDICTION')
    print(model.evaluate(testData, testTarget, verbose=1))
    y = model.predict(testData)
    
    score = ad.getScore(y, testTarget)
    auprc = ad.AUPRC(score, clabels)
    print('AUPRC : {}'.format(auprc))
    
    return model, y, auprc


def main_batch1(outlier):
    data, contaminedData, clabels = buildDataset(outlier=outlier, contamination=0.4)
    
#     #### Scale between [-1,1] for tanh activation
#     trainData = 2*trainData-1
#     testData = 2*testData-1
#     trainTarget = 2*trainTarget-1
#     testTarget = 2*testTarget-1
#     ### Supress above block if activation is sigmoid
    
    dim = 3
    model = build_lstm(input_shape=(None, dim), n_hidden1=200, n_hidden2=dim*10,
                       activation='sigmoid', optimizer=Adam(), loss='mse') 

    model, loss, val_loss = train_batch1(model, train=data, pred_len=10, epochs=25, val_split=0.2)
    model.save('lstm_ad_{}.h5'.format(outlier))
    
    print('\nPREDICTION')
    y, target, evloss = predict_batch1(model, train=data, test=contaminedData, pred_len=10)
    print('Evaluation loss : {}'.format(evloss))
    
    y = np.array(y).reshape(-1,30)
    target = np.array(target).reshape(-1,30)

    score = ad.getScore(y, target)
    auprc = ad.AUPRC(score, clabels)
    print('AUPRC : {}'.format(auprc))
    
    return model, y, auprc


if __name__=='___main__':
    main('Brush_teeth')
#     main_batch1('Comb_hair')
