#! /usr/bin/env bash

lang="julia"

model_path="/mnt/storage/ai/models/llm/deepseek-coder-1.3b-base"
ds_path="/mnt/storage/ai/data/ct_dataset/${lang}"
output_dir="results/deepseek-coder_data/${lang}"

python -m scripts.visualize_data \
	--model_name_or_path "${model_path}" \
	--dataset_name_or_path "${ds_path}" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 1000 \
	--max_eval_samples 20 \
	--max_test_samples 20 \
	--chunk_size 0 \
	--train_text_max_length 256 \
	--train_target_max_length 256 \
	--train_completions_only 1 \
	--valid_text_max_length 512 \
	--valid_target_max_length 512 \
	--test_text_max_length 1024 \
	--test_target_max_length 1024 \
	--output_dir "${output_dir}"
