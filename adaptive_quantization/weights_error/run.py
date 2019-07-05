import tensorflow as tf
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.varreplace import remap_variables
import argparse
import os
import numpy as np

cnt_relu_layer = 0
cnt_weights_layer = 0

id_target_quant_layer = -1
q_lambda = -1
q_delta = -1
num_quant_levels = -1
list_name_weights = [
    "conv1-conv2d-kernel",
    "conv2-conv2d-kernel",
    "conv3-conv2d-kernel",
    "fully_connected-fully_connected-kernel"]

HEIGHT = 1
WIDTH = 128
DEPTH = 9
NUM_CLASSES = 6


def read_all_lines(filename):
    with open(filename) as f:
        data = f.readlines()
    return data


def load_wei_codebook(path):
    data = read_all_lines(path)
    cluster = []
    for eachline in data[0:]:
        c = float(eachline)
        cluster.append(c)

    return cluster[0], cluster[1]


def weights_name_check(testing_name):
    if ("conv" in testing_name) and ("/kernel" in testing_name) and ("Momentum" not in testing_name) and (
            "shortcut" not in testing_name):
        return 1
    if ("fully_connected/kernel" in testing_name) and ("Momentum" not in testing_name):
        return 1
    return 0


def quant_wei_uni_dead_zone(x):
    if weights_name_check(x.name) == 1:
        global cnt_weights_layer

        print('go to layer %d (%d)' % (cnt_weights_layer, id_target_quant_layer))

        if cnt_weights_layer != id_target_quant_layer:
            cnt_weights_layer += 1
            return x

        print('quantizing layer %d ###########' % (cnt_weights_layer))
        q_k = int((num_quant_levels - 1) / 2)
        tensor_zeros = tf.multiply(x, 0.0)

        quant_values_abs = tf.math.abs(x)
        quant_values_signs = tf.math.sign(x)

        quant_values_divided_floored = tf.floor(tf.divide(tf.subtract(quant_values_abs, q_lambda), q_delta))
        quant_values_divided_floored_clipped = tf.clip_by_value(quant_values_divided_floored, 0, q_k - 1)
        quant_values = tf.multiply(
            tf.add(tf.multiply(quant_values_divided_floored_clipped, q_delta), q_lambda + q_delta / 2.0),
            quant_values_signs)

        condition = tf.less(tf.math.abs(x), q_lambda)
        quant_results = tf.where(condition, tensor_zeros, quant_values)
        cnt_weights_layer += 1

        return quant_results

    else:
        return x


def build_model(input_quant_wei_layer, intput_quant_wei_lambda, input_quant_wei_delta, input_quant_wei_levels):
    global id_target_quant_layer
    global q_lambda
    global q_delta
    global num_quant_levels

    id_target_quant_layer = input_quant_wei_layer
    q_lambda = intput_quant_wei_lambda
    q_delta = input_quant_wei_delta
    num_quant_levels = input_quant_wei_levels

    with tf.name_scope('main_params'):
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with remap_variables(quant_wei_uni_dead_zone), tf.variable_scope('conv1') as scope:
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

    with remap_variables(quant_wei_uni_dead_zone), tf.variable_scope('conv2') as scope:
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

    with remap_variables(quant_wei_uni_dead_zone), tf.variable_scope('conv3') as scope:
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

    with remap_variables(quant_wei_uni_dead_zone), tf.variable_scope('fully_connected') as scope:
        flat = tf.layers.flatten(relu3)
        logits = tf.layers.dense(inputs=flat, units=NUM_CLASSES, name=scope.name, use_bias=False,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))

    y_pred_cls = tf.argmax(logits, axis=1)
    gtlabel = tf.one_hot(label_y, NUM_CLASSES)
    # LOSS AND OPTIMIZER
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=gtlabel))
    loss = cross_entropy_loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-3).minimize(loss,
                                                                                               global_step=global_step)

    # PREDICTION AND ACCURACY CALCULATION
    def get_eval_op(preds, labels):
        correct_prediction = tf.equal(preds, labels)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    eval_op = get_eval_op(y_pred_cls, label_y)

    return loss, optimizer, eval_op, global_step, learning_rate


def summary_op(datapart):
    tf.summary.scalar(datapart + '-loss', loss)
    tf.summary.scalar(datapart + '-eval', eval_op)
    return tf.summary.merge_all()


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
        return dataset.map(self.parser)  # use padded_batch method if padding needed

    def preprocess(self, image):
        # MEAN_IMAGE = tf.constant([0.1], dtype=tf.float32)
        # STD_IMAGE = tf.constant([0.4], dtype=tf.float32)
        # image = (image - MEAN_IMAGE) / STD_IMAGE

        return image


def build_quantization():
    wei_quant_layer = int(args.wei_quant_layer)
    wei_prune_ratio = int(args.wei_prune_ratio)
    wei_quant_levels = int(args.wei_quant_levels)

    path_wei_codebooks = '../variables/weights_quantized_paras/' + list_name_weights[wei_quant_layer] + '_' \
                         + str(wei_prune_ratio) + '_' + str(wei_quant_levels) + '.cb'
    wei_quant_lambda, wei_quant_delta = load_wei_codebook(path_wei_codebooks)

    return wei_quant_layer, wei_prune_ratio, wei_quant_levels, wei_quant_lambda, wei_quant_delta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wei_quant_layer', help='wei quantization layer')
    parser.add_argument('--wei_prune_ratio', help='wei prune ratio')
    parser.add_argument('--wei_quant_levels', help='wei quantization levels')
    parser.add_argument('--logits', help='original logits output file')

    args = parser.parse_args()
    isTraining = False

    WEIGHT_DECAY = 1e-4

    SAVE_PATH = "../uci_har/model/"
    train_f, test_f = ['../uci_har/har-%s.tfrecords' % i for i in ['train', 'eval']]

    epochs = 60
    batch_size = 100  # when batch_size can't be divided by nDatas, like 56,
    tf.set_random_seed(21)

    # training dataset
    nDatasTrain = 7352
    har_train = HARDataSet(subset='train')
    dataset_train = har_train.get_dataset(train_f)
    dataset_train = dataset_train.batch(batch_size)  # make sure repeat is ahead batch
    # print dataset_train
    nBatchs = nDatasTrain // batch_size

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

    wei_quant_layer, wei_prune_ratio, wei_quant_levels, wei_quant_lambda, wei_quant_delta = build_quantization()
    loss, optimizer, eval_op, global_step, learning_rate = build_model(wei_quant_layer, wei_quant_lambda,
                                                                       wei_quant_delta, wei_quant_levels)
    init = tf.global_variables_initializer()

    summary_op_train = summary_op('train')
    summary_op_val = summary_op('val')

    # whether to restore or not
    ckpt_nm = 'cnn-ckpt'
    saver = tf.train.Saver()  # defaults to save all variables, using dict {'x':x,...} to save specified ones.

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
        if ckpt and ckpt.model_checkpoint_path:  # ckpt.model_checkpoint_path means the latest ckpt
            if restore_step == 'latest':
                ckpt_f = tf.train.latest_checkpoint(SAVE_PATH)
                start_step = int(ckpt_f.split('-')[-1]) + 1
            else:
                ckpt_f = SAVE_PATH + ckpt_nm + '-' + restore_step
            print('loading wgt file: ' + ckpt_f)
            saver.restore(sess, ckpt_f)

            selected_tensor_names = ['fully_connected/fully_connected/MatMul']
            selected_tensors = [tf.get_default_graph().get_tensor_by_name(name + ":0") for name in
                                selected_tensor_names]
            fc_out_predicted = sess.run(selected_tensors, feed_dict={handle: handle_train})[0]
            for i in xrange(1, nBatchs):
                fc_out_batch = sess.run(selected_tensors, feed_dict={handle: handle_train})[0]
                fc_out_predicted = np.concatenate((fc_out_predicted, fc_out_batch), axis=0)

            with open(args.logits, 'rb') as f:
                fc_out_origin = np.fromfile(args.logits, dtype=np.float32)
                fc_out_origin = np.reshape(fc_out_origin, (7300, 6))

            output_error = ((fc_out_origin - fc_out_predicted) ** 2).mean(axis=None)
            print('output error is %.16f' % output_error)

            output_path = 'results/output_error_' + str(wei_quant_layer) + '_' + str(wei_prune_ratio) + '_' + str(
                wei_quant_levels)
            file_results = open(output_path, 'w')
            file_results.write(str(output_error) + '\n')
            file_results.close()
