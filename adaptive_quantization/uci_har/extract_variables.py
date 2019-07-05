
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
    def __init__(self, subset='train'):
        self.subset = subset

    def parser(self, serialized_example):
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


def build_model():
    with tf.name_scope('main_params'):
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=data_x,
            filters=32,
            kernel_size=[1, 64],
            padding='VALID',
            use_bias=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        )
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
        bn3 = tf.layers.batch_normalization(conv3, training=isTraining)
        relu3 = tf.nn.relu(bn3)

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.layers.flatten(relu3)
        logits = tf.layers.dense(inputs=flat, units=NUM_CLASSES, name=scope.name, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))

    y_pred_cls = tf.argmax(logits, axis=1)
    gtlabel = tf.one_hot(label_y, NUM_CLASSES)
    # LOSS AND OPTIMIZER
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=gtlabel))
    loss = cross_entropy_loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-3).minimize(loss, global_step=global_step)

    # PREDICTION AND ACCURACY CALCULATION
    def get_eval_op(preds, labels):
        correct_prediction = tf.equal(preds, labels)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    eval_op = get_eval_op(y_pred_cls, label_y)

    return loss, optimizer, eval_op, global_step, learning_rate, optimizer


def summary_op(datapart):
    tf.summary.scalar(datapart + '-loss', loss)
    tf.summary.scalar(datapart + '-eval', eval_op)
    return tf.summary.merge_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--batch_extract', action='store_true', help='extract batch for quantized parameter learning')
    parser.add_argument('--full_extract', action='store_true', help='extract last layer for error computation')

    args = parser.parse_args()
    isTraining = False

    WEIGHT_DECAY = 1e-4

    SAVE_PATH = "./model/"
    train_f, test_f = ['./har-%s.tfrecords' % i for i in ['train', 'eval']]

    epochs = 60
    batch_size = 100  # when batch_size can't be divided by nDatas, like 56,
    tf.set_random_seed(21)

    # training dataset
    nDatasTrain = 7352
    har_train = HARDataSet(subset='train')
    dataset_train = har_train.get_dataset(train_f)
    dataset_train = dataset_train.batch(batch_size) # make sure repeat is ahead batch
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
    iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)
    data_x, label_y = iterator.get_next()
    # print x, y
    # x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, DEPTH])

    loss, optimizer, eval_op, global_step, learning_rate, optimizer = build_model()
    init = tf.global_variables_initializer()

    summary_op_train = summary_op('train')
    summary_op_val = summary_op('val')

    # whether to restore or not
    ckpt_nm = 'cnn-ckpt'
    saver = tf.train.Saver() # defaults to save all variables, using dict {'x':x,...} to save specified ones.

    restore_step = 'latest'
    start_step = 0
    best_loss = 1e6
    best_step = 0

    def lr_func(epoch):
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
        handle_train, handle_test = sess.run([x.string_handle() for x in [iter_train, iter_test]])

        ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
            if restore_step == 'latest':
                ckpt_f = tf.train.latest_checkpoint(SAVE_PATH)
                start_step = int(ckpt_f.split('-')[-1]) + 1
            else:
                ckpt_f = SAVE_PATH+ckpt_nm+'-'+restore_step
            print('loading wgt file: ' + ckpt_f)
            saver.restore(sess, ckpt_f)

            ops_file = './model/ops.txt'
            all_op_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
            with open(ops_file, 'wb') as f:
                for item in all_op_names:
                    f.write("%s\n" % item)

            if args.batch_extract:
                selected_tensor_names = ['conv1/conv2d/kernel', 'conv1/Relu',
                                         'conv2/conv2d/kernel', 'conv2/Relu',
                                         'conv3/conv2d/kernel', 'conv3/Relu',
                                         'fully_connected/fully_connected/kernel']
                selected_tensors = [tf.get_default_graph().get_tensor_by_name(name + ":0") for name in
                                    selected_tensor_names]
                conv1_w, conv1_out, conv2_w, conv2_out, conv3_w, conv3_out, fc_w = \
                    sess.run(selected_tensors, feed_dict={handle: handle_train})

                print conv1_w.shape, conv1_out.shape, conv2_w.shape, conv2_out.shape, conv3_w.shape, conv3_out.shape
                print fc_w.shape

                conv1_w.astype('float32').tofile(os.path.join('../variables/weights/',
                                                              '-'.join(selected_tensor_names[0].split('/'))))
                conv1_out.astype('float32').tofile(os.path.join('../variables/activations/',
                                                                '-'.join(selected_tensor_names[1].split('/'))))
                conv2_w.astype('float32').tofile(os.path.join('../variables/weights/',
                                                              '-'.join(selected_tensor_names[2].split('/'))))
                conv2_out.astype('float32').tofile(os.path.join('../variables/activations/',
                                                                '-'.join(selected_tensor_names[3].split('/'))))
                conv3_w.astype('float32').tofile(os.path.join('../variables/weights/',
                                                              '-'.join(selected_tensor_names[4].split('/'))))
                conv3_out.astype('float32').tofile(os.path.join('../variables/activations/',
                                                                '-'.join(selected_tensor_names[5].split('/'))))
                fc_w.astype('float32').tofile(os.path.join('../variables/weights/',
                                                           '-'.join(selected_tensor_names[6].split('/'))))

            elif args.full_extract:
                selected_tensor_names = ['fully_connected/fully_connected/MatMul']
                selected_tensors = [tf.get_default_graph().get_tensor_by_name(name + ":0") for name in
                                    selected_tensor_names]

                fc_out = sess.run(selected_tensors, feed_dict={handle: handle_train})[0]
                for i in xrange(1, nBatchs):
                    fc_out_batch = sess.run(selected_tensors, feed_dict={handle: handle_train})[0]
                    fc_out = np.concatenate((fc_out, fc_out_batch), axis=0)

                print fc_out.shape
                fc_out.astype('float32').tofile(os.path.join('../variables/',
                                                             '-'.join(selected_tensor_names[0].split('/'))))
