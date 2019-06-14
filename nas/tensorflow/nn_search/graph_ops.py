# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:24:32 2018

@author: lile
"""

import numpy as np
import os
import time
import sys
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from PIL import Image
sys.path.append("..")
from object_detection.utils import label_map_util
from kitti import kitti_data
from matplotlib import pyplot as plt
from google.protobuf import text_format

job_name = 'cifar10_nns'

#%% extract from inference graph
PATH_TO_CKPT = 'outputs/' + job_name + '/inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    all_op_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

with open(os.path.join('outputs', job_name, job_name + '_op_names.txt'), 'w') as fp:
    for i in xrange(len(all_op_names)):
        fp.write("%s\n" % all_op_names[i].encode("UTF-8"))

#%% extract from graph.pbtxt
PATH_TO_CKPT = 'outputs/' + job_name + '/graph.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    text_format.Merge(fid.read(), graph_def)
    tf.import_graph_def(graph_def, name='')

with detection_graph.as_default():
    all_op_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

with open(os.path.join('outputs', job_name, job_name + '_op_names_graph.txt'), 'w') as fp:
    for i in xrange(len(all_op_names)):
        fp.write("%s\n" % all_op_names[i].encode("UTF-8"))
