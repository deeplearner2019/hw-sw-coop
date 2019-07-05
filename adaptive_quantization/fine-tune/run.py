import tensorflow as tf
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.varreplace import remap_variables
import argparse
import os
import numpy as np

inf = 100000000
num_quant_layer_wei = 4
num_quant_layer_act = 3
wei_quant_lambda = []
wei_quant_delta = []
wei_quant_levels = []
act_quant_delta = []
act_quant_levels = []


cnt_wei_layer = 0
cnt_act_layer = 0


list_name_weights = [
    "conv1-conv2d-kernel",
    "conv2-conv2d-kernel",
    "conv3-conv2d-kernel",
    "fully_connected-fully_connected-kernel"]

list_name_acts = [
    "conv1-Relu",
    "conv2-Relu",
    "conv3-Relu"]

HEIGHT = 1
WIDTH = 128
DEPTH = 9
NUM_CLASSES = 6


def read_all_lines(filename):
    with open(filename) as f:
        data = f.readlines()
    return data


def load_codebook_wei(codebook_file):
    data = read_all_lines(codebook_file)
    cluster = []

    for eachline in data[0:]:
        c = float(eachline)
        cluster.append(c)

    return cluster[0], cluster[1]


def load_codebook_act(codebook_file):
    data = read_all_lines(codebook_file)
    cluster = []

    for eachline in data[0:]:
        c = float(eachline)
        cluster.append(c)

    return cluster[1]


def load_bits_allocation(input_file):
    dead_zones = []
    quant_levels = []

    with open(input_file) as f:
        for line in f:
            a, b = [int(x) for x in line.split()]
            dead_zones.append(a)
            quant_levels.append(b)

    return dead_zones, quant_levels


def load_quantization_configs(dir_bit_allocation=[], path_wei_codebooks=[], path_act_codebooks=[]):
    all_weights_names = list_name_weights
    dead_zones, quant_levels = load_bits_allocation(dir_bit_allocation)

    for i in range(num_quant_layer_wei):
        dir_wei_codebooks = path_wei_codebooks + '/' + all_weights_names[i] + '_' + str(dead_zones[i]) + '_' + str(
            quant_levels[i]) + '.cb'
        quant_lambda, quant_delta = load_codebook_wei(dir_wei_codebooks)
        wei_quant_lambda.append(quant_lambda)
        wei_quant_delta.append(quant_delta)
        wei_quant_levels.append(quant_levels[i])

    for i in range(num_quant_layer_act):
        dir_act_codebooks = path_act_codebooks + '/' + list_name_acts[i] + '_0_' + str(
            quant_levels[i + num_quant_layer_wei]) + '.cb'
        if inf == quant_levels[i + num_quant_layer_wei]:
            quant_delta = -1
        else:
            quant_delta = load_codebook_act(dir_act_codebooks)
        act_quant_delta.append(quant_delta)
        act_quant_levels.append(quant_levels[i + num_quant_layer_wei])

    return


def weights_name_check(testing_name):
    if ("conv" in testing_name) and ("kernel" in testing_name) and ("Momentum" not in testing_name) and (
            "shortcut" not in testing_name):
        return 1
    if ("fully_connected/kernel" in testing_name) and ("Momentum" not in testing_name):
        return 1
    return 0


def quant_wei_uni_dead_zone(x):
    if weights_name_check(x.name) == 1:
        global cnt_wei_layer
        global wei_quant_lambda
        global wei_quant_delta
        global wei_quant_levels
        print('the id of weights %d' % (cnt_wei_layer))
        num_quant_levels = wei_quant_levels[cnt_wei_layer]
        print('finish the id of weights')
        q_lambda = wei_quant_lambda[cnt_wei_layer]
        q_delta = wei_quant_delta[cnt_wei_layer]

        q_k = int((num_quant_levels - 1) / 2)

        @tf.custom_gradient
        def _quant_wei_uni_dead_zone(x):
            tensor_zeros = tf.multiply(x, 0.0)
            quant_values_abs = tf.abs(x)
            quant_values_signs = tf.sign(x)

            quant_values_divided_floored = tf.floor(tf.divide(tf.subtract(quant_values_abs, q_lambda), q_delta))
            quant_values_divided_floored_clipped = tf.clip_by_value(quant_values_divided_floored, 0,
                                                                    tf.cast(q_k - 1, tf.float32))
            quant_values = tf.multiply(
                tf.add(tf.multiply(quant_values_divided_floored_clipped, q_delta), q_lambda + q_delta / 2.0),
                quant_values_signs)

            condition = tf.less(tf.abs(x), q_lambda)
            quant_results = tf.where(condition, tensor_zeros, quant_values)
            global cnt_wei_layer
            cnt_wei_layer += 1
            return quant_results, lambda dy: dy

        return _quant_wei_uni_dead_zone(x)

    else:
        return x


def quant_act_uni(x, base, num_quant_levels):
    if inf == num_quant_levels:
        return x

    @tf.custom_gradient
    def _quant_act_uni(x):
        x_divided = tf.divide(x, base)
        x_divided_rounded = tf.round(x_divided)
        x_divided_rounded_clipped = tf.clip_by_value(x_divided_rounded, clip_value_min=0,
                                                     clip_value_max=tf.cast(num_quant_levels - 1, tf.float32))
        x_quant_values = tf.multiply(x_divided_rounded_clipped, base)

        return x_quant_values, lambda dy: dy

    return _quant_act_uni(x)


def build_model(wei_quant_lambda_input, wei_quant_delta_input, wei_quant_levels_input,
                act_quant_delta_input, act_quant_levels_input):
    global cnt_act_layer
    global wei_quant_lambda
    global wei_quant_delta
    global wei_quant_levels
    global act_quant_delta
    global act_quant_levels

    wei_quant_lambda = wei_quant_lambda_input
    wei_quant_delta = wei_quant_delta_input
    wei_quant_levels = wei_quant_levels_input
    act_quant_delta = act_quant_delta_input
    act_quant_levels = act_quant_levels_input

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
        relu1 = quant_act_uni(relu1, act_quant_delta[cnt_act_layer], act_quant_levels[cnt_act_layer])
        cnt_act_layer += 1

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
        relu2 = quant_act_uni(relu2, act_quant_delta[cnt_act_layer], act_quant_levels[cnt_act_layer])
        cnt_act_layer += 1

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
        relu3 = quant_act_uni(relu3, act_quant_delta[cnt_act_layer], act_quant_levels[cnt_act_layer])
        cnt_act_layer += 1

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    # quantization
    parser.add_argument('--bit_allocation', help='direction of bit allocation')
    parser.add_argument('--dir_weight_codebooks', help='path of codebooks of weights')
    parser.add_argument('--dir_activation_codebooks', help='path of codebooks of activations')
    args = parser.parse_args()

    isTraining = True
    if args.eval:
        isTraining = False

    WEIGHT_DECAY = 1e-4

    SAVE_PATH = "./models/scratch/model_" + str(float(args.bit_allocation.split('_')[-1].split('.')[0])/10) + '/'
    train_f, test_f = ['../uci_har/har-%s.tfrecords' % i for i in ['train', 'eval']]

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
    iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)
    data_x, label_y = iterator.get_next()
    # print x, y
    # x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, DEPTH])

    load_quantization_configs(args.bit_allocation, args.dir_weight_codebooks, args.dir_activation_codebooks)
    loss, optimizer, eval_op, global_step, learning_rate = build_model(wei_quant_lambda, wei_quant_delta,
                                                                       wei_quant_levels, act_quant_delta,
                                                                       act_quant_levels)
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
                print('loading wgt file: ' + ckpt_f)
                saver.restore(sess, ckpt_f)

                val_loss = 0.0
                val_eval = 0.0
                for i in range(0, nTestBatchs):
                    cur_val_loss, cur_val_eval, summary = sess.run([loss, eval_op, summary_op_val],
                                                                   feed_dict={handle: handle_test})
                    val_loss += cur_val_loss
                    val_eval += cur_val_eval
                print ('loss val %0.5f, acc val %.5f' % (val_loss/nTestBatchs, val_eval/nTestBatchs))
        else:
            summary_wrt = tf.summary.FileWriter(SAVE_PATH, sess.graph)
            start_epoch = start_step * batch_size // nDatasTrain
            for e in range(start_epoch, epochs):
                train_loss = 0.0
                train_eval = 0.0
                for i in range(0, nBatchs):
                    i_global, cur_loss, cur_train_eval, summary, _ = \
                        sess.run([global_step, loss, eval_op, summary_op_train, optimizer],
                                 feed_dict={handle: handle_train, learning_rate: lr_func(e+1)})
                    train_loss += cur_loss
                    train_eval += cur_train_eval

                # log to stdout and eval validation set
                saver.save(sess, SAVE_PATH + ckpt_nm, global_step=i_global) # save variables
                summary_wrt.add_summary(summary, global_step=i_global)

                print ('epoch %3d, learning_rate %.6f: loss %.5f, acc %.5f' % (e+1, lr_func(e+1),
                                                                               train_loss/nBatchs, train_eval/nBatchs))