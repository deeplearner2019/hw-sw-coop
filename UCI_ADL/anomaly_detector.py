import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def fitNormal(data):
    mean = np.mean(data, axis=0)
    cov = 0
    for d in data:
        cov += np.dot((d-mean).reshape(len(d), 1), (d-mean).reshape(1, len(d)))
    cov /= len(data)
    return mean, cov

def getScore(data, target):
    error = abs(data-target)
    mean, cov = fitNormal(error)
    score = []
    covInv = np.linalg.pinv(cov)
    for e in error:
        d = np.dot(e-mean, covInv)
        d = np.dot(e, (e-mean).T)
        score.append(d)
    return score

def AUPRC(score, label):
    prec, rec, th = precision_recall_curve(np.array(label), np.array(score))
    return auc(rec, prec)