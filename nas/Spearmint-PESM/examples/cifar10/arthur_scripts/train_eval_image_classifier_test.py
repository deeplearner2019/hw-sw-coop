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
"""Generic training script that trains a model using a given dataset."""



import train_eval_image_classifier


if __name__ == '__main__':
   params = {}
   params['b0_i1'] = [-1]
   params['b0_i2'] = [-1]
   params['b0_o1'] = [0]
   params['b0_o2'] = [0]

   for i in xrange(1, 5):
      params['b{}_i1'.format(i)] = [-2]
      params['b{}_i2'.format(i)] = [-2]
      params['b{}_o1'.format(i)] = [0]
      params['b{}_o2'.format(i)] = [0]
   
   job_id = 238
   print(train_eval_image_classifier.main(job_id, params))
