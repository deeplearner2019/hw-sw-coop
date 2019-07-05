#!/usr/bin/env bash

gpu_id=1

for i in {2..3..1}
do
	for j in {10..90..10}
	do

		for k in {3..31..2}
		do
		    output_path="results/output_error_"$i"_"$j"_"$k
            if [[ -f ${output_path} ]]; then
                echo "File already exists!" ${output_path}
                continue
            fi
			CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
			--wei_quant_layer $i \
			--wei_prune_ratio $j \
			--wei_quant_levels $k \
            --logits ../variables/fully_connected-fully_connected-MatMul
		done

		for k in {41..501..10}
		do
		    output_path="results/output_error_"$i"_"$j"_"$k
            if [[ -f ${output_path} ]]; then
                echo "File already exists!" ${output_path}
                continue
            fi

			CUDA_VISIBLE_DEVICES=${gpu_id}  python ./run.py \
			--wei_quant_layer $i \
			--wei_prune_ratio $j \
			--wei_quant_levels $k \
            --logits ../variables/fully_connected-fully_connected-MatMul
		done

		for k in {601..1001..100}
		do
		    output_path="results/output_error_"$i"_"$j"_"$k
            if [[ -f ${output_path} ]]; then
                echo "File already exists!" ${output_path}
                continue
            fi

			CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
			--wei_quant_layer $i \
			--wei_prune_ratio $j \
			--wei_quant_levels $k \
            --logits ../variables/fully_connected-fully_connected-MatMul
		done

		for k in 2048 4096 8192 16384 32768 65536
		do
		    output_path="results/output_error_"$i"_"$j"_"$k
            if [[ -f ${output_path} ]]; then
                echo "File already exists!" ${output_path}
                continue
            fi

			CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
			--wei_quant_layer $i \
			--wei_prune_ratio $j \
			--wei_quant_levels $k \
            --logits ../variables/fully_connected-fully_connected-MatMul
		done
	done
done
