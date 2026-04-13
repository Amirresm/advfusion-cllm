# AdvFusion

This repository contains the implementation of experiments for the paper on **AdvFusion**, a parameter-efficient fine-tuning (PEFT) method for multilingual knowledge transfer in code language models (Code-LLMs).

---

## Overview

This work studies how different PEFT methods, particularly fusion-based approaches, behave in multilingual software engineering tasks. While prior work suggested that adversarial fusion (AdvFusion) can improve over standard AdapterFusion, our findings show that its effectiveness is **context-dependent**, varying across tasks, programming languages, and model architectures.

This repository enables a systematic evaluation of:

- **Standard PEFT methods** (e.g., LoRA, bottleneck adapters, Compacter)
- **AdapterFusion** for combining language-specific adapters
- **AdvFusion**, which modifies fusion training dynamics to encourage cross-language knowledge utilization

This repository provides a framework to understand **how fusion-based approaches perform** in code-related tasks such as:

- Commit Message Generation (CMG)
- Code Generation (CG)
- Code Translation (CT)

---

# Run

## Environment

This project requires **Python 3.12+** and a GPU-enabled environment.

### Recommended (uv)

```bash
uv sync
source .venv/bin/activate
```

### Alternative (venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Training

### Individual Adapters

Train a single adapter (or PEFT model) for a specific task/language:

```bash
python -m scripts.train \
  --model_name_or_path <model> \
  --dataset_name_or_path <dataset_dir> \
  --train_file train.jsonl \
  --validation_file valid.jsonl \
  --test_file test.jsonl \
  --do_train --do_eval \
  --output_dir results/<experiment>
```

Supported configurations include:

- LoRA (`--lib peft --peft_method lora`)
- AdapterHub bottleneck adapters (`--lib adp --peft_method seq_bn`)
- Compacter

---

### AdapterFusion

Train a fusion layer over multiple pre-trained adapters:

```bash
python -m scripts.train_fusion \
  --model_name_or_path <model> \
  --adapter_path_list <adapter1> <adapter2> ... \
  --dataset_name_or_path <dataset_dir> \
  --do_train --do_eval \
  --output_dir results/<fusion>
```

**Requirements:**

- All adapters must be trained on the same base model
- Adapters must be AdapterHub-style (not generic LoRA checkpoints)

---

### AdvFusion

Train adversarial fusion over multiple adapters:

```bash
python -m scripts.train_advf \
  --model_name_or_path <model> \
  --adapter_path_list <adapter1> <adapter2> ... \
  --target_adapter_path <target_adapter> \
  --dataset_name_or_path <dataset_dir> \
  --do_train --do_eval \
  --output_dir results/<advfusion>
```

AdvFusion training consists of two stages:

1. Training with the target adapter suppressed
2. Reintroducing the target adapter and training fusion

---

## Outputs

Each run produces:

```
output_dir/
├── config.yaml
├── adapter/ or adapter_fusion/
└── results/
    ├── evaluation_results.json
    ├── *_metrics.json
    ├── *_samples.jsonl
    └── resource_usage.json
```

---

## Supported Models

Adversarial fusion is supported for:

- Llama-2 / Llama-3
- Qwen-2.5
- Gemma-2
- DeepSeek-Coder

Model type is inferred from the model name/path. If needed:

```bash
--model_type <type>
```

---

## PEFT Configs

PEFT configurations are defined in `src/peft/configs.py`.
You can customize or extend PEFT methods by editing this file.

---

## Other Commands

Inspect available options:

```bash
python -m scripts.train --help
python -m scripts.train_fusion --help
python -m scripts.train_advf --help
```

Visualize preprocessing:

```bash
python -m scripts.visualize_data ...
```

---

# Data

## Datasets

The experiments use the following datasets:

### Commit Message Generation (CMG)

- Repo: https://github.com/bigcode-project/octopack
- Data: https://huggingface.co/datasets/bigcode/commitpackft

### Code Generation (CG)

- Repo: https://github.com/ntunlp/xCodeEval
- Data: https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval

### Code Translation (CT)

- Repo: https://github.com/WeixiangYAN/CodeTransOcean
- Data: https://huggingface.co/datasets/WeixiangYan/CodeTransOcean

### Expected Format

Datasets should be provided as:

```
train.jsonl
valid.jsonl
test.jsonl
```

---

## Preprocessing

The preprocessing pipeline:

1. Load dataset (JSON/JSONL)
2. Convert rows into `(input, target)`
3. Tokenize input and target
4. Concatenate tokens
5. Mask input tokens if `--train_completions_only`
6. Apply truncation or chunking

Each dataset type is handled via custom processors in:

```
src/dataset/custom_processors.py
```

---

# Extending the Repository

### Adding a Dataset

1. Register dataset type in:

```
src/dataset/utils.py
```

2. Implement a processor in:

```
src/dataset/custom_processors.py
```

Each processor must return:

```
{
  "input": ...,
  "target": ...
}
```
