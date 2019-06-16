import os
import tensorflow as tf
import argparse
from quantize_graph import experimental_create_training_graph, experimental_create_eval_graph
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HEIGHT = 1
WIDTH = 128
DEPTH = 9
NUM_CLASSES = 6

class HARDataSet(object):
    """Cifar10 data set.

    Described by http://www.cs.toronto.edu/~kriz/cifar.html.
    """
    def __init__(self, subset='train'):
        self.subset = subset

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([DEPTH * HEIGHT * WIDTH], tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = features['image']

        # Reshape from [depth * height * width] to [height, width, depth].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [HEIGHT, WIDTH, DEPTH]), [0, 1, 2]),
            tf.float32)
        label = tf.cast(features['label'], tf.int64)

        # Custom preprocessing.
        image = self.preprocess(image)

        return image, label

    def get_dataset(self, fname):
        dataset = tf.data.TFRecordDataset(fname)
        return dataset.map(self.parser) # use padded_batch method if padding needed

    def preprocess(self, image):
        # MEAN_IMAGE = tf.constant([0.1], dtype=tf.float32)
        # STD_IMAGE = tf.constant([0.4], dtype=tf.float32)
        # image = (image - MEAN_IMAGE) / STD_IMAGE

        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    args = parser.parse_args()
    isTraining = True
    if args.eval:
        isTraining = False

    WEIGHT_DECAY = 1e-4

    #FIXME
    bitwidth = 7
    delayQuant = 1000
    SAVE_PATH = "./model-8bit-v2-inputQuant/"
    train_f, test_f = ['./har-%s-quant.tfrecords' % i for i in ['train', 'eval']]


    epochs = 60
    batch_size = 100  # when batch_size can't be divided by nDatas, like 56,
    tf.set_random_seed(21)

    # training dataset
    nDatasTrain = 7352
    har_train = HARDataSet(subset='train')
    dataset_train = har_train.get_dataset(train_f)
    dataset_train = dataset_train.repeat(epochs).shuffle(nDatasTrain).batch(batch_size) # make sure repeat is ahead batch
    # print dataset_train
    nBatchs = nDatasTrain//batch_size

    # test dataset
    nDatasTest = 2947
    har_test = HARDataSet(subset='eval')
    dataset_test = har_test.get_dataset(test_f)
    # print dataset_test
    dataset_test = dataset_test.batch(nDatasTest)
    nTestBatchs = 1

    # make dataset iterator
    iter_train = dataset_train.make_one_shot_iterator()
    iter_test = dataset_test.make_one_shot_iterator()

    # make feedable iterator, i.e. iterator placeholder
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, \
                                                   dataset_train.output_types, dataset_train.output_shapes)
    data_x, label_y = iterator.get_next()
    # print x, y
    # x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, DEPTH])

    with tf.name_scope('main_params'):
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        phase = tf.placeholder(tf.bool, name='phase')
        drop_prob = tf.placeholder('float')

    with tf.variable_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=data_x,
            filters=32,
            kernel_size=[1, 64],
            padding='VALID',
            use_bias=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        )
        #FIXME
        bn1 = tf.layers.batch_normalization(conv1, training=isTraining)
        relu1 = tf.nn.relu(bn1)
        # pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    with tf.variable_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(
            inputs=relu1,
            filters=64,
            kernel_size=[1, 32],
            padding='VALID',
            use_bias=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        )
        #FIXME
        bn2 = tf.layers.batch_normalization(conv2, training=isTraining)
        relu2 = tf.nn.relu(bn2)
        # pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    with tf.variable_scope('conv3') as scope:
        conv3 = tf.layers.conv2d(
            inputs=relu2,
            filters=128,
            kernel_size=[1, 16],
            padding='VALID',
            use_bias=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        )
        #FIXME
        bn3 = tf.layers.batch_normalization(conv3, training=isTraining)
        relu3 = tf.nn.relu(bn3)

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.layers.flatten(relu3)
        # fc = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))
        # drop = tf.layers.dropout(fc, rate=drop_prob, name='drop0', training=isTraining)
        logits = tf.layers.dense(inputs=flat, units=NUM_CLASSES, name=scope.name, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))

    y_pred_cls = tf.argmax(logits, axis=1)
    gtlabel = tf.one_hot(label_y, NUM_CLASSES)
    # LOSS AND OPTIMIZER
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=gtlabel))
    loss = cross_entropy_loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    #FIXME
    g = tf.get_default_graph()
    if isTraining:
        experimental_create_training_graph(input_graph=g, weight_bits=bitwidth, activation_bits=bitwidth, quant_delay=delayQuant)
    else:
        experimental_create_eval_graph(input_graph=g, weight_bits=bitwidth, activation_bits=bitwidth)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-3).minimize(loss, global_step=global_step)

    # PREDICTION AND ACCURACY CALCULATION
    def get_eval_op(preds, labels):
        correct_prediction = tf.equal(preds, labels)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    eval_op = get_eval_op(y_pred_cls, label_y)

    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    # summary
    def summary_op(datapart = 'train'):
        tf.summary.scalar(datapart + '-loss', loss)
        tf.summary.scalar(datapart + '-eval', eval_op)
        return tf.summary.merge_all()
    summary_op_train = summary_op()
    summary_op_val = summary_op('val')

    # whether to restore or not
    ckpt_nm = 'cnn-ckpt'
    saver = tf.train.Saver() # defaults to save all variables, using dict {'x':x,...} to save specified ones.

    #FIXME
    restore_step = 'latest'
    start_step = 0
    best_loss = 1e6
    best_step = 0

    def lr(epoch):
        learning_rate = 1e-2
        if epoch > 90:
            learning_rate *= 0.5e-3
        elif epoch > 70:
            learning_rate *= 1e-3
        elif epoch > 50:
            learning_rate *= 1e-2
        elif epoch > 30:
            learning_rate *= 1e-1
        return learning_rate

    with tf.Session() as sess:
        sess.run(init)
        handle_train, handle_test = sess.run( \
            [x.string_handle() for x in [iter_train, iter_test]])

        if args.eval:
            ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
                if restore_step == 'latest':
                    ckpt_f = tf.train.latest_checkpoint(SAVE_PATH)
                    start_step = int(ckpt_f.split('-')[-1]) + 1
                else:
                    ckpt_f = SAVE_PATH+ckpt_nm+'-'+restore_step
                print('loading wgt file: '+ ckpt_f)
                saver.restore(sess, ckpt_f)

                val_loss = 0.0
                val_eval = 0.0
                for i in range(0, nTestBatchs):
                    cur_val_loss, cur_val_eval, summary = sess.run([cross_entropy_loss, eval_op, summary_op_val], \
                                                                              feed_dict={handle: handle_test, drop_prob: 0.0, phase: False})
                    val_loss += cur_val_loss
                    val_eval += cur_val_eval
                print ('loss val %0.5f, acc val %.5f' % (val_loss/nTestBatchs, val_eval/nTestBatchs))

        else:
            # ele = sess.run(iterator.get_next(), feed_dict={handle: handle_test})
            # print ele[0][1, :, :, :]
            # print ele[0].shape

            summary_wrt = tf.summary.FileWriter(SAVE_PATH, sess.graph)
            start_epoch = start_step * batch_size // nDatasTrain
            for e in range(start_epoch, epochs):
                train_loss = 0.0
                train_eval = 0.0
                for i in range(0, nBatchs):
                    _, i_global, cur_loss, cur_train_eval, summary, cur_input, cur_conv1, cur_conv2, cur_conv3 \
                        = sess.run([optimizer, global_step, loss, eval_op, summary_op_train, data_x, conv1, conv2, conv3], \
                                                    feed_dict={handle: handle_train, drop_prob: 0.5, learning_rate: lr(e+1), phase: True})
                    # print i_logits
                    # print(cur_input.shape)
                    # print(cur_conv1.shape)
                    # print(cur_conv2.shape)
                    # print(cur_conv3.shape)
                    # exit()
                    train_loss += cur_loss
                    train_eval += cur_train_eval

                # log to stdout and eval validation set
                saver.save(sess, SAVE_PATH + ckpt_nm, global_step=i_global) # save variables
                summary_wrt.add_summary(summary, global_step=i_global)

                print ('epoch %3d, learning_rate %.6f: loss %.5f, acc %.5f' % (e+1, lr(e+1), \
                                                    train_loss/nBatchs, train_eval/nBatchs))
