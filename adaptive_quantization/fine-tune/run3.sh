#!/usr/bin/env bash

gpu_id=2
for i in {81..120..1}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
    --bit_allocation ../variables/bit_allocations/bit_allocation_$i.txt \
	--dir_weight_codebooks ../variables/weights_quantized_paras \
	--dir_activation_codebooks ../variables/activations_quantized_paras \
	|& tee results/logs_$i.txt
done