#!/bin/bash

model_name=$1
PATH_TO_HACONE='/home/lile/Projects/git_repo/hacone'

python $PATH_TO_HACONE/tensorflow/nn_search/export_inference_graph_movidius.py --job_name=cifar10_movidius --name_job=$model_name --output_file=$PATH_TO_HACONE/models_trained/$model_name/inference_graph.pb --PATH_TO_HACONE=$PATH_TO_HACONE

python $PATH_TO_HACONE/tensorflow/nn_search/freeze_graph_16.py --input_graph=$PATH_TO_HACONE/models_trained/$model_name/inference_graph.pb --input_checkpoint=$PATH_TO_HACONE/models_trained/$model_name/model.ckpt-28125 --input_binary=true --output_graph=$PATH_TO_HACONE/models_trained/$model_name/frozen_graph.pb --output_node_names=CifarNet/Predictions/Reshape_1

make all MODEL_DIR=$PATH_TO_HACONE/models_trained/$model_name/



