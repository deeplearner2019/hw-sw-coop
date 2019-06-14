# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the CIFAR-10 model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)

def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions.

  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def cifarnet(images, candidate, N, F, num_classes=10, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='CifarNet'):
  """Creates a variant of the CifarNet model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = cifarnet.cifarnet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    candidate: a 20-d vector for network architecture representation. [i1, i2, op1, op2]
    num_classes: the number of classes in the dataset. If 0 or None, the logits
      layer is omitted and the input features to the logits layer are returned
      instead.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the input to the logits layer if num_classes
      is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  end_points = {}
  assert len(candidate) == 20

  with tf.variable_scope(scope, 'CifarNet', [images]):
    net = images
    cell_id = -2
    end_points['cell_{}'.format(cell_id)] = net
    cell_id += 1

    end_points['cell_{}'.format(cell_id)] = net
    cell_id += 1

    for i in xrange(N):
        net = build_cell(end_points, cell_id, candidate, F, stride = 1, scope = 'Normal_cell_{}'.format(cell_id),is_training=is_training)
        end_points['cell_{}'.format(cell_id)] = net
        cell_id += 1

    net = build_cell(end_points, cell_id, candidate, F, stride = 2, scope = 'Reduction_cell_{}'.format(cell_id),is_training=is_training)
    end_points['cell_{}'.format(cell_id)] = net
    cell_id += 1

    F = F * 2

    for i in xrange(N):
        net = build_cell(end_points, cell_id, candidate, F, stride = 1, scope = 'Normal_cell_{}'.format(cell_id),is_training=is_training)
        end_points['cell_{}'.format(cell_id)] = net
        cell_id += 1

    net = build_cell(end_points, cell_id, candidate, F, stride = 2, scope = 'Reduction_cell_{}'.format(cell_id),is_training=is_training)
    end_points['cell_{}'.format(cell_id)] = net
    cell_id += 1

    F = F * 2

    for i in xrange(N):
        net = build_cell(end_points, cell_id, candidate, F, stride = 1, scope = 'Normal_cell_{}'.format(cell_id),is_training=is_training)
        end_points['cell_{}'.format(cell_id)] = net
        cell_id += 1

    net = tf.reduce_mean(net, [1, 2], name='global_pool', keep_dims=False)

    if not num_classes:
      return net, end_points
    logits = slim.fully_connected(net, num_classes,
                                  biases_initializer=tf.zeros_initializer(),
                                  weights_initializer=trunc_normal(1/192.0),
                                  weights_regularizer=None,
                                  activation_fn=None,
                                  scope='logits',
                                  trainable = is_training)

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
cifarnet.default_image_size = 32

operations = {}
operations['0'] = '3x3_depth_conv'
operations['1'] = '5x5_depth_conv'
operations['2'] = '7x7_depth_conv'
operations['3'] = 'identity'
operations['4'] = '3x3_avg_pool'
operations['5'] = '3x3_max_pool'
operations['6'] = '3x3_dilated_conv'
operations['7'] = '1x7_7x1'

def apply_operation(net, op_name, num_outputs, stride = 1, scope = None, is_training=False):
   with tf.variable_scope(scope):
    if op_name == '3x3_depth_conv':
       net = slim.separable_conv2d(net, num_outputs, [3, 3], 1, stride = stride, activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, normalizer_params={'is_training':is_training}, scope = op_name)
       return net
    if op_name == '5x5_depth_conv':
       net = slim.separable_conv2d(net, num_outputs, [5, 5], 1, stride = stride, activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, normalizer_params={'is_training':is_training}, scope = op_name )
       return net
    if op_name == '7x7_depth_conv':
       net = slim.separable_conv2d(net, num_outputs, [7, 7], 1, stride = stride, activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, normalizer_params={'is_training':is_training}, scope = op_name)
       return net
    if op_name == 'identity':
       if net.get_shape().as_list()[3] == num_outputs:
           net = subsample(net, stride, scope = op_name)
       else:
           net = slim.conv2d(net, num_outputs, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None, scope = op_name)
       return net
    if op_name == '3x3_avg_pool':
       if net.get_shape().as_list()[3] != num_outputs:
          net = slim.conv2d(net, num_outputs, [1, 1], stride=1,
                             normalizer_fn=None, activation_fn=None, scope = 'depth_match')
       net = slim.avg_pool2d(net, [3,3], stride = stride, padding = 'SAME', scope = op_name)
       return net
    if op_name == '3x3_max_pool':
       if net.get_shape().as_list()[3] != num_outputs:
          net = slim.conv2d(net, num_outputs, [1, 1], stride=1,
                             normalizer_fn=None, activation_fn=None, scope = 'depth_match')
       net = slim.max_pool2d(net, [3,3], stride = stride, padding = 'SAME', scope = op_name)
       return net
    if op_name == '3x3_dilated_conv':
       if stride == 1:
          rate = 2
       else:
          rate = 1 # in reduction cell, dilated conv degrade to normal conv
       net = slim.conv2d(net, num_outputs, [3,3], stride = stride, rate = rate, activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, normalizer_params={'is_training':is_training}, scope = op_name + '_rate_{}'.format(rate))
       return net
    if op_name == '1x7_7x1':
       net = slim.conv2d(net, num_outputs, [1, 7], stride = 1, activation_fn = None, normalizer_fn = None, scope = 'conv1x7')
       net = slim.conv2d(net, num_outputs, [7, 1], stride = stride, activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, normalizer_params={'is_training':is_training}, scope = 'conv7x1')
       return net



def build_cell(end_points, cell_id, candidate, num_outputs, stride = 1, scope = None, is_training=False):
    input_points = {}

    input_points['-2'] = end_points['cell_{}'.format(cell_id - 2)]
    input_points['-1'] = end_points['cell_{}'.format(cell_id - 1)]
    B = len(candidate) // 4

    unused = [1] * B
    output_height = []
    with tf.variable_scope(scope):
       for i in xrange(B):
          i1 = candidate[4*i]
          i2 = candidate[4*i + 1]

          assert i1 < i
          assert i2 < i

          if i1 >= 0:
             unused[i1] = 0
          if i2 >= 0:
             unused[i2] = 0

          op1 = operations[str(candidate[4*i + 2])]
          op2 = operations[str(candidate[4*i + 3])]

          if stride == 2:
             if i1 < 0:
                o1 = apply_operation(input_points[str(i1)], op1, num_outputs, stride = 2, scope = 'block_{}_op1'.format(i),is_training=is_training)
             else:
                o1 = apply_operation(input_points[str(i1)], op1, num_outputs, stride = 1, scope = 'block_{}_op1'.format(i),is_training=is_training) # if input of current block is the output of previous block, don't do stride 2

             if i2 < 0:
                o2 = apply_operation(input_points[str(i2)], op2, num_outputs, stride = 2, scope = 'block_{}_op2'.format(i),is_training=is_training)
             else:
                o2 = apply_operation(input_points[str(i2)], op2, num_outputs, stride = 1, scope = 'block_{}_op2'.format(i),is_training=is_training)
          else:
             o1 = apply_operation(input_points[str(i1)], op1, num_outputs, stride = stride, scope = 'block_{}_op1'.format(i),is_training=is_training)
             o2 = apply_operation(input_points[str(i2)], op2, num_outputs, stride = stride, scope = 'block_{}_op2'.format(i),is_training=is_training)

          print("output dimension of o1 for cell {} block {} with input {} and operation {}:".format(cell_id, i, i1, op1))
          print(o1.get_shape().as_list())
          print("output dimension of o2 for cell {} block {} with input {} and operation {}:".format(cell_id, i, i2, op2))
          print(o2.get_shape().as_list())

          if stride == 1: # for normal cell, the output width/height can be different
              if o1.get_shape().as_list()[1] > o2.get_shape().as_list()[1]:
                 o1 = slim.max_pool2d(o1, [3,3], stride = 2, padding = 'SAME', scope = 'block_{}_op1_resize'.format(i))
              elif o1.get_shape().as_list()[1] <  o2.get_shape().as_list()[1]:
                 o2 = slim.max_pool2d(o2, [3,3], stride = 2, padding = 'SAME', scope = 'block_{}_op2_resize'.format(i))

          input_points[str(i)] = o1 + o2
          output_height.append(input_points[str(i)].get_shape().as_list()[1])

       if stride == 1: # for normal cell, the output width/height of different blocks can be different
          min_height = min(output_height)
          for i in xrange(B):
              if input_points[str(i)].get_shape().as_list()[1] > min_height:
                 input_points[str(i)] = slim.max_pool2d(input_points[str(i)], [3,3], stride = 2, padding = 'SAME', scope = 'block_{}_resize'.format(i))

       output = tf.concat([input_points[str(i)] for i in xrange(B) if unused[i] == 1], axis = 3, name = scope + '_unused_concat')
       output = slim.max_pool2d(output,[1,1],stride=1,scope='fake_pool')
    return output



def cifarnet_arg_scope(weight_decay=0.004):
  """Defines the default cifarnet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d],
      weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
      activation_fn=tf.nn.relu):
    with slim.arg_scope(
        [slim.fully_connected],
        biases_initializer=tf.constant_initializer(0.1),
        weights_initializer=trunc_normal(0.04),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu) as sc:
      return sc
