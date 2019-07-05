#!/usr/bin/env bash

gpu_id=3

for i in {0..2..1}
do
	for k in {3..31..2}
        do
            output_path="results/act_output_error_"$i"_"$k
            if [[ -f ${output_path} ]]; then
                echo "File already exists!" ${output_path}
                continue
            fi
            CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
            --actQuantLayer $i \
            --actQuantLevels $k
        done

	for k in {41..501..10}
        do
            output_path="results/act_output_error_"$i"_"$k
            if [[ -f ${output_path} ]]; then
                echo "File already exists!" ${output_path}
                continue
            fi
            CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
            --actQuantLayer $i \
            --actQuantLevels $k
        done

    for k in {601..1001..100}
        do
            output_path="results/act_output_error_"$i"_"$k
            if [[ -f ${output_path} ]]; then
                echo "File already exists!" ${output_path}
                continue
            fi
            CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
            --actQuantLayer $i \
            --actQuantLevels $k
        done


    for k in 2048 4096 8192 16384 32768 65536
        do
            output_path="results/act_output_error_"$i"_"$k
            if [[ -f ${output_path} ]]; then
                echo "File already exists!" ${output_path}
                continue
            fi
            CUDA_VISIBLE_DEVICES=${gpu_id} python ./run.py \
            --actQuantLayer $i \
            --actQuantLevels $k
        done
done
