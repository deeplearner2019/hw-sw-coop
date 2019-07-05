#!/usr/bin/env bash

gpu_id=0

for i in {20..160..10}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
    --bit_allocation ../variables/bit_allocations/bit_allocation_$i.txt \
	--dir_weight_codebooks ../variables/weights_quantized_paras \
	--dir_activation_codebooks ../variables/activations_quantized_paras \
	|& tee results/logs_${i}_train.txt

	CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
	--eval \
    --bit_allocation ../variables/bit_allocations/bit_allocation_$i.txt \
	--dir_weight_codebooks ../variables/weights_quantized_paras \
	--dir_activation_codebooks ../variables/activations_quantized_paras \
	|& tee results/logs_${i}_eval.txt
done