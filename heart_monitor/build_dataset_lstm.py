from utils import config
from imutils import paths
import random
import shutil
import cv2
import numpy as np
import os

## Read spectrograms of training set
## as greyscale array and resize to shape (64,64)
imagePaths = list(paths.list_images(config.TRAIN_PATH))
random.seed(42)
random.shuffle(imagePaths)
X_train = []
Y_train = np.zeros((len(imagePaths), 2))
for i, image in enumerate(imagePaths):
    image_class = image.split('/')[-2]
    if image_class=='normal':
        Y_train[i,0] = 1
    else:
        Y_train[i,1] = 1
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X_train.append(res)
X_train, Y_train = np.array(X_train), np.array(Y_train)


## Read spectrograms of validation set
## as greyscale array and resize to shape (64,64)
imagePaths = list(paths.list_images(config.VAL_PATH))
random.seed(42)
random.shuffle(imagePaths)
X_val = []
Y_val = np.zeros((len(imagePaths), 2))
for i, image in enumerate(imagePaths):
    image_class = image.split('/')[-2]
    if image_class=='normal':
        Y_val[i,0] = 1
    else:
        Y_val[i,1] = 1
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X_val.append(res)
X_val, Y_val = np.array(X_val), np.array(Y_val)


## Read spectrograms of test set
## as greyscale array and resize to shape (64,64)
imagePaths = list(paths.list_images(config.TEST_PATH))
random.seed(42)
random.shuffle(imagePaths)
X_test = []
Y_test = np.zeros((len(imagePaths), 2))
for i, image in enumerate(imagePaths):
    image_class = image.split('/')[-2]
    if image_class=='normal':
        Y_test[i,0] = 1
    else:
        Y_test[i,1] = 1
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(im, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X_test.append(res)
X_test, Y_test = np.array(X_test), np.array(Y_test)


## Scale datasets
xmax = 0
xmin = 255
for image in X_train:
    if xmin > np.min(image):
        xmin = np.min(image)
    if xmax < np.max(image):
        xmax = np.max(image)

X_train_sc = np.zeros(X_train.shape)
for i, image in enumerate(X_train):
    X_train_sc[i, :, :] = 2*((image-xmin)/(xmax-xmin))-1

X_val_sc = np.zeros(X_val.shape)
for i, image in enumerate(X_val):
    X_val_sc[i, :, :] = 2*((image-xmin)/(xmax-xmin))-1
    
X_test_sc = np.zeros(X_test.shape)
for i, image in enumerate(X_test):
    X_test_sc[i, :, :] = 2*((image-xmin)/(xmax-xmin))-1


## Save arrays to npz format
np.savez(os.path.join(config.BASE_PATH, 'data_heart'),
         X_train=X_train_sc, X_val=X_val_sc, X_test=X_test_sc,
         Y_train=Y_train, Y_val=Y_val, Y_test=Y_test)