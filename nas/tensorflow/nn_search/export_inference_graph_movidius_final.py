#!/home/anne-maelle/hacone/tensorflow-py27/bin/python
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Saves out a GraphDef containing the architecture of the model.

To use it, run something like this, with a model name defined by slim:

bazel build tensorflow_models/research/slim:export_inference_graph
bazel-bin/tensorflow_models/research/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb

To run the resulting graph in C++, you can look at the label_image sample code:

bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

import sys
#sys.path.append('/mnt/data_b/anne-maelle/hacone/tensorflow/nn_search')
#sys.path.append('/mnt/data_b/anne-maelle/hacone/tensorflow/slim')


from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nn_search.nets import nets_factory
import os

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

def main(model_name, N, F):
  PATH_TO_HACONE = '/home/lile/Projects/git_repo/hacone'
  final_model_name = model_name + '_N{}_F{}'.format(N, F)
  output_file = PATH_TO_HACONE + '/outputs/final_models/{}/inference_graph.pb'.format(final_model_name)
 

  with open('{}/jobs/job{}.txt'.format(PATH_TO_HACONE, model_name), 'r') as fp:
      data = json.load(fp)
  job_id = data['job']
  params = data['params']
  params = json.loads(params)

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default() as graph:
    dataset = dataset_factory.get_dataset('cifar10', 'test', '/home/lile/dataset/cifar10')

    candidate = []
    print("main")
    #print(tf.app.flags.FLAGS.train_dir)

    for i in xrange(0, 5):
        candidate.append(params['b{}_i1'.format(i)])
        candidate.append(params['b{}_i2'.format(i)])
        candidate.append(params['b{}_o1'.format(i)])
        candidate.append(params['b{}_o2'.format(i)])
    print(candidate)


    network_fn = nets_factory.get_network_fn(
        'cifarnet', candidate, N, F,
        num_classes=(dataset.num_classes - 0),
        is_training=False)

    image_size =  network_fn.default_image_size
    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                 shape=[None, image_size,
                                        image_size, 3])
    network_fn(placeholder)
    graph_def = graph.as_graph_def()
    with gfile.GFile(output_file, 'wb') as f:
      f.write(graph_def.SerializeToString())


if __name__ == '__main__':
  model_name = sys.argv[1]
  N = int(sys.argv[2])
  F = int(sys.argv[3])
  main(model_name, N, F)
