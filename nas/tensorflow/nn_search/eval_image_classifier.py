# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nn_search.nets import nets_factory
from preprocessing import preprocessing_factory
import os
import json
import time

def eval_model(model_name):
  slim = tf.contrib.slim

  print("eval model")
  PATH_TO_HACONE_LOCAL = '/home/lile/Projects/git_repo/hacone'

  with open(PATH_TO_HACONE_LOCAL + '/jobs/job{}.txt'.format(model_name)) as fp:
    data = json.load(fp)

    job_id = data['job']
    params = data['params']
    params = json.loads(params)

  candidate = []

  for i in xrange(0, 5):
    candidate.append(params['b{}_i1'.format(i)])
    candidate.append(params['b{}_i2'.format(i)])
    candidate.append(params['b{}_o1'.format(i)])
    candidate.append(params['b{}_o2'.format(i)])

  N = 2
  F = 24   

  dataset_dir = '/home/lile/dataset/cifar10_val'
  batch_size = 100
  output_dir = os.path.join(PATH_TO_HACONE_LOCAL,'models_trained', model_name) 
  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        'cifar10', 'val', dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        'cifarnet', candidate, N, F,
        num_classes=(dataset.num_classes - 0),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * batch_size,
        common_queue_min= batch_size)
    [image, label] = provider.get(['image', 'label'])


    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = 'cifarnet'
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

   
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)


    variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    
    num_batches = math.ceil(dataset.num_samples / float(batch_size))

    checkpoint_path = output_dir
    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
      checkpoint_path = checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    final_op = [names_to_values['Accuracy']] #top1 accuracy to return
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    start_time = time.time()
    a = slim.evaluation.evaluate_once(
        master='',
        checkpoint_path=checkpoint_path,
        logdir=output_dir,
        session_config=config,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        final_op = final_op,
        variables_to_restore=variables_to_restore)
    duration = time.time() - start_time
    print('________________________________')
    print('duration :' + str(duration))
    print('________________________________')

    print(a)
    return duration
