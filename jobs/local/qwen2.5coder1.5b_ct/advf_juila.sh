#!/usr/bin/env bash

source "$(dirname "$0")/../_setup.sh"

echo "Starting job on '$MACHINE' at $(date) in project root: $PROJECT_ROOT"

OUTPUT_DIR="$PROJECT_ROOT/results/jobs/qwen2.5coder1.5b_ct/advf_julia"
mkdir -p "$OUTPUT_DIR"

lang="julia"
model_path="$STORAGE_ROOT/ai/models/llm/Qwen2.5-Coder-1.5B"
ds_path="$STORAGE_ROOT/ai/data/ct_dataset/${lang}"

target_adapter_path="/home/amirreza/projects/ubc/advfusion/jobs/results/jobs/qwen2.5coder1.5b_ct/adp_julia"

adapter_path_list=(
	"/home/amirreza/projects/ubc/advfusion/jobs/results/jobs/qwen2.5coder1.5b_ct/adp_julia"
	"/home/amirreza/projects/ubc/advfusion/jobs/results/jobs/qwen2.5coder1.5b_ct/adp_r"
)

benchmark_dataset_name_or_path="/home/amirreza/projects/ubc/multipl_e_ct/data/processed/ct_bench_dataset_go_julia.jsonl"

python -m scripts.train_advf \
	--model_name_or_path "${model_path}" \
	--q "4bit" \
	--adapter_path_list "${adapter_path_list[@]}" \
	--target_adapter_path "$target_adapter_path" \
	--dataset_name_or_path "${ds_path}" \
	--train_file train.jsonl \
	--validation_file valid.jsonl \
	--test_file test.jsonl \
	--max_train_samples 9999999 \
	--max_eval_samples 64 \
	--max_test_samples 64 \
	--epochs 1 \
	--chunk_size 0 \
	--train_text_max_length 1536 \
	--train_target_max_length 1024 \
	--train_max_length 2048 \
	--train_completions_only False \
	--do_train 1 \
	--learning_rate 1e-4 \
	--train_batch_size 2 \
	--gradient_accumulation_steps 1 \
	--do_eval 1 \
	--logging_steps 0.05 \
	--eval_steps 0.2 \
	--eval_batch_size 2 \
	--gen_pre_train_max_samples 20 \
	--gen_batch_size 32 \
	--benchmark_dataset_name_or_path "${benchmark_dataset_name_or_path}" \
	--benchmark_dataset_type "CodeTranslationBench" \
	--benchmark_do_sample 1 \
	--benchmark_temperature 0.3 \
	--benchmark_top_p 0.9 \
	--benchmark_top_k 50 \
	--benchmark_n_per_sample 1 \
	--output_dir "${OUTPUT_DIR}"
