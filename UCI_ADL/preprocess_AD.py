import numpy as np
np.random.seed(1)

from os.path import join
from pathlib import Path
import pickle
from time import time

from keras.preprocessing.sequence import pad_sequences

from parse_data import ActId


def getData(activity, source='data/HMP_Dataset'):
    file = join(source, activity, '{}_{}_all.pkl'.format(ActId[activity], activity))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def buildDataset(outlier, contamination, ActId=ActId):
    '''
    Return 3 arrays :
    - a training set containing only normal timeseries
    - a test set containing normal and anomalous timeseries
    - an array of label for the test set with 1 if the element is an anomaly, 0 otherwise
    
    Parameters
    ----------
    outlier : class to consider as anomalous
    contamination : ratio of anomalous samples in the test set
    ActId : dict containing the different classes
    '''

    data = np.array([])
    for activity in ActId:
        if activity != outlier:
            actData = getData(activity)
            data = np.concatenate((data, actData), axis=0)
    np.random.shuffle(data)
    
    contaminedData= np.array(getData(outlier))
    np.random.shuffle(contaminedData)
    
    # Manually setting a threshold (here, 50) of anomalous samples to avoid having a large test set
    n_anomaly = len(contaminedData) if len(contaminedData)<50 else 50
    n_inliers = round(n_anomaly*(1-contamination)/contamination)

    label = [1]*n_anomaly+[0]*n_inliers
    contaminedData = np.concatenate((contaminedData[:n_anomaly], data[-n_inliers:]), axis=0)
    data = data[:-n_inliers]
    
    return data, contaminedData, label
    

def MinMaxSc(train, test):
    """
    Depthwise scaling of the timeseries between [0,1]
    """
    
    for i in range(train.shape[2]):
        amax = np.amax(train[:,:,i])
        amin = np.amin(train[:,:,i])
        train[:, :, i] = (train[:, :, i]-amin) / (amax-amin)
        test[:, :, i] = (test[:, :, i]-amin) / (amax-amin)

    return train, test

    
def prepareTS(train, test, pred_len):
    """
    Pre pad with 0 value and scale timeseries.
    
    If pred_len > 0, separate the datasets into two for prediction :
    - a training set of timeseries of length (n_timesteps - pred_len)
    - a target set of timeseries of length pred_len
    """
    
    trainShape = [array.shape[0] for array in train]
    testShape = [array.shape[0] for array in test]
    maxlen = max(max(trainShape), max(testShape))
    train = pad_sequences(train, maxlen=maxlen, value=0.).astype(np.float64)
    test = pad_sequences(test, maxlen=maxlen, value=0.).astype(np.float64)
    train, test = MinMaxSc(train, test)
    
    if pred_len > 0:
        trainData = train[:,:-pred_len,:]
        trainTarget = train[:,-pred_len:,:].reshape(-1, pred_len*3)
        testData = test[:,:-pred_len,:]
        testTarget = test[:,-pred_len:,:].reshape(-1, pred_len*3)
        return trainData, trainTarget, testData, testTarget
    
    else:
        return train, test
    
    
def getMinMax(data):
    minmax = [(0,0), (0,0), (0,0)]
    for array in data:
        for i in range(array.shape[-1]):
            amin = minmax[i][0]
            amax = minmax[i][1]
            mn = np.amin(array[:,i])
            mx = np.amax(array[:,i])
            if amin > mn:
                amin = mn
            if amax < mx:
                amax = mx
            minmax[i] = (amin, amax)
    return minmax


def MinMaxSc_batch1(data, minmax):
    scaled = np.empty(data.shape)
    for i in range(data.shape[-1]):
        amin = minmax[i][0]
        amax = minmax[i][1]
        scaled[:,i] = (data[:,i]-amin) / (amax-amin)
    return scaled
