import os
import sys
import numpy as np
import tensorflow as tf
# from settings import app
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data_train():
    if os.path.isfile('data/data_har.npz') == True:
        data = np.load('data/data_har.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
    return X_train, Y_train

def load_data_test():
    if os.path.isfile('data/data_har.npz') == True:
        data = np.load('data/data_har.npz')
        X_test = data['X_test']
        Y_test = data['Y_test']
    return X_test, Y_test

X_train, Y_train = load_data_train()
print(X_train.shape)
Y_train = [np.where(r==1)[0][0] for r in Y_train]
X_test, Y_test = load_data_test()
print(X_test.shape)
Y_test = [np.where(r==1)[0][0] for r in Y_test]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _write_tfrecord(X_data, Y_data, num_samples, output_filename):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for i in range(num_samples):
        try:
            # print(_float_feature(X_data[i,:,:].flatten().astype(np.float64)))
            # print(Y_data[i,:].astype(np.int32))
            # print(np.shape(X_data[i,:,:].reshape(-1)))
            # print(np.shape(int(Y_data[i])))
            feature = {
                'image': _float_feature(X_data[i,:,:].reshape(-1)),
                # 'image': _bytes_feature(tf.compat.as_bytes(encoded_image_string)),
                'label': _int64_feature(int(Y_data[i]))
            }
            tf_example = tf.train.Example(features = tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
        except Exception as inst:
            print(inst)
            pass
    writer.close()

def _write_sharded_tfrecord(X_data, Y_data, num_samples, is_training = True):
    output_filename = 'har-{0}-quant4.tfrecords'.format(
        'train' if is_training else 'eval'
    )
    _write_tfrecord(X_data, Y_data, num_samples, output_filename)


k = 8
min = -2
max = 2
n = float(2 ** k - 1)
X_train = np.clip(X_train, min, max)
X_train = np.round(X_train * n) / n
X_test = np.clip(X_test, min, max)
X_test = np.round(X_test * n) / n

print("Creating training shards")
# _write_sharded_tfrecord(X_train, Y_train, 7352, True)
# with open('train.txt', 'w') as out:
#     for fi in range(X_train.shape[0]):   #num_filter
#         np.savetxt(out, X_train[fi,:,:], fmt='%11.8f')
plt.figure()
plt.subplot(2, 1, 1)
print(np.mean(X_train), np.std(X_train), np.max(X_train), np.min(X_train))
plt.hist(X_train.reshape(-1), 100, color='r')


print("\nCreating test shards")
# _write_sharded_tfrecord(X_test, Y_test, 2947, False)
# with open('test.txt', 'w') as out:
#     for fi in range(X_test.shape[0]):   #num_filter
#         np.savetxt(out, X_test[fi,:,:], fmt='%11.8f')
# print("\n", flush = True)
plt.subplot(2, 1, 2)
print(np.mean(X_test), np.std(X_test), np.max(X_test), np.min(X_test))
plt.hist(X_test.reshape(-1), 100, color='r')

name = 'stat-8bit.png'
plt.savefig(name)
plt.close()