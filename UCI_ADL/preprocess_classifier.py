import numpy as np
np.random.seed(1)

from os.path import join
from pathlib import Path
import pickle
from time import time

from parse_data import ActId

def getData(activity, source='data/HMP_Dataset'):
    file = join(source, activity, '{}_{}_all.pkl'.format(ActId[activity], activity))
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def getSeq(timeseries, seq_step, seq_length):
    """
    Return an array of sliced timeseries of same length. The subsequences are obtained using sliding window.
    
    Parameters
    ----------
    seq_step : sliding window's step.
    seq_lenght : output length of the sliced timeseries.
    """
    
    n_seq = (timeseries.shape[0] - seq_length) // seq_step
    x = np.zeros((n_seq, seq_length, timeseries.shape[1]))
    for i in range(n_seq):
        x[i,:,:] = timeseries[i*seq_step:i*seq_step+seq_length, :]
    return x


def buildDataset(seq_step, seq_length, dim):
    """
    Return the dataset containing the subsequences of timeseries for each class and the corresponding labels.
    Each label is a vector of shape (n_class), with 1 if the subsequence belong to the class, 0 otherwise.
    
    Parameters
    ----------
    seq_step : sliding window's step.
    seq_lenght : output length of the sliced timeseries.
    dim : number of features of the timeseries.
    """
    
    sub_seq = np.zeros((1, seq_length, dim))
    labels = np.zeros((1, len(ActId)))

    for act in ActId:
        data = getData(act)
        for timeseries in data:
            sq = getSeq(timeseries, seq_step, seq_length)
            lb = np.zeros((len(sq), len(ActId)))
            lb[:, ActId[act]-1] = 1
            sub_seq = np.concatenate((sub_seq, sq), axis=0)
            labels = np.concatenate((labels, lb), axis=0)

    return sub_seq[1:], labels[1:]

    
def splitDataset(dataset, labels, split=0.8, shuffle=True):
    """
    Split dataset into train and test sets.
    
    Parameters
    ----------
    split : split ratio for train and test sets.
    shuffle : (default=True) whether the data are shuffled before the splitting.
    """
    
    if shuffle:
        indices = np.arange(dataset.shape[0])
        np.random.shuffle(indices)
        dataset, labels = dataset[indices], labels[indices]
    
    th = int(len(dataset)*split)
    train, test = dataset[:th], dataset[th:]
    train_lb, test_lb = labels[:th], labels[th:]
    
    return train, test, train_lb, test_lb


def MinMaxSc(train, test=None):
    """
    Depthwise scaling of the timeseries between [0,1]
    """
    
    for i in range(train.shape[2]):
        amax = np.amax(train[:,:,i])
        amin = np.amin(train[:,:,i])
        train[:, :, i] = (train[:, :, i]-amin) / (amax-amin)
        if test is not None:
            test[:, :, i] = (test[:, :, i]-amin) / (amax-amin)
            
    if test is not None:
        return train, test
    else:
        return train
