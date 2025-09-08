from dataclasses import dataclass
import json
import os

import torch
import simple_parsing
from rich import print as rprint

from src.advfusion.advfusion import (
    init_advfusion,
    load_advfusion,
    reload_advf_target_adapter,
)
from src.advfusion.args import AdvfArgs
from src.dataset.args import DatasetArgs
from src.dataset.utils import DatasetType
from src.generation.args import GenArgs
from src.generation.generation import generate_raw_samples
from src.model.args import ModelArgs
from src.model.model import init_model, init_tokenizer
from src.dataset.dataset import load_raw_dataset, preprocess_dataset
from src.model.utils import ModelType
from src.train.args import TrainArgs
from src.train.trainer import get_trainer
from src.utils.args import parse_args
from src.utils.resource_logger import ResourceLogger


@dataclass
class Args(simple_parsing.Serializable):
    model: ModelArgs
    advf: AdvfArgs
    dataset: DatasetArgs
    train: TrainArgs
    gen: GenArgs

    output_dir: str = (
        ""  # The output directory where the model predictions and checkpoints will be written.
    )


def main():
    args = parse_args(Args)
    assert type(args.model.model_type) is ModelType
    assert type(args.dataset.dataset_type) is DatasetType

    output_dir = os.path.realpath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_dir = os.path.join(output_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        args.dump_yaml(f)

    resource_logger = ResourceLogger(
        save_path=os.path.join(output_dir, "results", "resource_usage.json")
    )
    resource_logger.clear_cuda(hard=True)
    resource_logger.record("init", current=True)

    resource_logger.clear_cuda()
    model, model_dtype = init_model(
        args.model.model_name_or_path,
        quantization_mode=args.model.q,
    )
    resource_logger.record_baseline()
    tokenizer = init_tokenizer(args.model.model_name_or_path, model)

    raw_dataset = load_raw_dataset(
        args.dataset.dataset_name_or_path,
        dataset_type=args.dataset.dataset_type,
        train_file=args.dataset.train_file,
        test_file=args.dataset.test_file,
        validation_file=args.dataset.validation_file,
        max_train_samples=args.dataset.max_train_samples,
        max_validation_samples=args.dataset.max_eval_samples,
        max_test_samples=args.dataset.max_test_samples,
    )

    if args.advf.preload_advf_from is not None:
        print(f"Loading AdvFusion model from {args.advf.preload_advf_from}")
        fusion_name = load_advfusion(
            model,
            args.model.model_type,
            target_adapter_path=args.advf.target_adapter_path,
            preload_path=args.advf.preload_advf_from,
            dtype=model_dtype,
        )
    else:
        fusion_name = init_advfusion(
            model,
            args.model.model_type,
            adapter_path_list=args.advf.adapter_path_list,
            target_adapter_path=args.advf.target_adapter_path,
            dtype=model_dtype,
        )

    if (args.gen.gen_pre_train_max_samples or 0) > 0:
        print("Generating raw samples before training...")
        with torch.inference_mode():
            resource_logger.clear_cuda()
            generate_raw_samples(
                model,
                tokenizer,
                raw_dataset,
                args.gen.gen_pre_train_max_samples,
                batch_size=args.gen.gen_batch_size,
                max_new_tokens=args.gen.generate_max_new_tokens,
                do_sample=args.gen.gen_do_sample,
                temperature=args.gen.gen_temperature,
                top_k=args.gen.gen_top_k,
                top_p=args.gen.gen_top_p,
                n_per_sample=args.gen.gen_n_per_sample,
                save_path=os.path.join(results_dir, "pre_trained"),
            )
            resource_logger.record("pre_trained_generate")

    train_dataset = None
    if "train" in raw_dataset:
        train_dataset = preprocess_dataset(
            raw_dataset,
            "train",
            tokenizer=tokenizer,
            output_dir=results_dir,
            only_completion=args.dataset.train_completions_only,
            chunk_size=args.dataset.chunk_size,
            text_max_length=args.dataset.train_text_max_length,
            target_max_length=args.dataset.train_target_max_length,
            debug=args.dataset.debug,
        )
    eval_dataset = None
    if "validation" in raw_dataset:
        eval_dataset = preprocess_dataset(
            raw_dataset,
            "validation",
            tokenizer=tokenizer,
            output_dir=results_dir,
            only_completion=True,
            chunk_size=0,
            text_max_length=args.dataset.valid_text_max_length
            or args.dataset.train_text_max_length,
            target_max_length=args.dataset.valid_target_max_length
            or args.dataset.train_target_max_length,
            debug=args.dataset.debug,
        )
    test_dataset = None
    if "test" in raw_dataset:
        test_dataset = preprocess_dataset(
            raw_dataset,
            "test",
            tokenizer=tokenizer,
            output_dir=results_dir,
            only_completion=True,
            chunk_size=0,
            text_max_length=args.dataset.test_text_max_length
            or args.dataset.train_text_max_length,
            target_max_length=args.dataset.test_target_max_length
            or args.dataset.train_target_max_length,
            debug=args.dataset.debug,
        )

    training_metrics = {}
    if train_dataset and args.train.do_train:
        print("Training the model...")
        trainer = get_trainer(
            model=model,
            tokenizer=tokenizer,
            peft_lib="adp",
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            train_batch_size=args.train.train_batch_size,
            eval_batch_size=args.train.eval_batch_size,
            epochs=args.train.epochs,
            learning_rate=args.train.learning_rate,
            warmup_ratio=args.train.warmup_ratio,
            optim=args.train.optim,
            logging_steps=args.train.logging_steps,
            eval_steps=args.train.eval_steps,
            eval_accumulation_steps=args.train.eval_accumulation_steps,
            gradient_accumulation_steps=args.train.gradient_accumulation_steps,
            gradient_checkpointing=args.train.gradient_checkpointing,
            max_length=args.dataset.train_max_length,
        )
        resource_logger.clear_cuda()
        results_pre = trainer.train()
        resource_logger.record("train_excluded")
        training_metrics["train_excluded"] = results_pre.metrics

        if (args.gen.gen_pre_train_max_samples or 0) > 0:
            with torch.inference_mode():
                print("Generating raw samples after excluded training...")
                resource_logger.clear_cuda()
                generate_raw_samples(
                    model,
                    tokenizer,
                    raw_dataset,
                    args.gen.gen_pre_train_max_samples,
                    batch_size=args.gen.gen_batch_size,
                    max_new_tokens=args.gen.generate_max_new_tokens,
                    do_sample=args.gen.gen_do_sample,
                    temperature=args.gen.gen_temperature,
                    top_k=args.gen.gen_top_k,
                    top_p=args.gen.gen_top_p,
                    n_per_sample=args.gen.gen_n_per_sample,
                    save_path=os.path.join(results_dir, "excluded"),
                )
                resource_logger.record("excluded_generate")

        reload_advf_target_adapter(
            model,
            target_adapter_path=args.advf.target_adapter_path,
            dtype=model_dtype,
        )
        model.train_adapter_fusion(fusion_name)
        resource_logger.clear_cuda()
        results = trainer.train()
        resource_logger.record("train")
        training_metrics["train"] = results.metrics

        fusion_path = os.path.join(args.output_dir, "adapter_fusion")
        print(f"Saving adapter fusion to {fusion_path}")
        model.save_adapter_setup(fusion_path, fusion_name)

        if args.train.do_eval:
            with torch.inference_mode():
                if eval_dataset:
                    print("Evaluating on the validation set...")
                    resource_logger.clear_cuda()
                    eval_results = trainer.evaluate(
                        eval_dataset=eval_dataset,  # type: ignore
                        metric_key_prefix="eval",
                    )
                    resource_logger.record("eval")
                    training_metrics["eval"] = eval_results

                if test_dataset:
                    print("Evaluating on the test set...")
                    resource_logger.clear_cuda()
                    test_results = trainer.evaluate(
                        eval_dataset=test_dataset,  # type: ignore
                        metric_key_prefix="test",
                    )
                    resource_logger.record("test")
                    training_metrics["test"] = test_results

    else:
        if (args.gen.gen_pre_train_max_samples or 0) > 0:
            with torch.inference_mode():
                print("Generating raw samples after excluded training...")
                resource_logger.clear_cuda()
                generate_raw_samples(
                    model,
                    tokenizer,
                    raw_dataset,
                    args.gen.gen_pre_train_max_samples,
                    batch_size=args.gen.gen_batch_size,
                    max_new_tokens=args.gen.generate_max_new_tokens,
                    do_sample=args.gen.gen_do_sample,
                    temperature=args.gen.gen_temperature,
                    top_k=args.gen.gen_top_k,
                    top_p=args.gen.gen_top_p,
                    n_per_sample=args.gen.gen_n_per_sample,
                    save_path=os.path.join(results_dir, "excluded"),
                )
                resource_logger.record("excluded_generate")

        reload_advf_target_adapter(
            model,
            target_adapter_path=args.advf.target_adapter_path,
            dtype=model_dtype,
        )

    for split in training_metrics:
        rprint(f"[bold green]Metrics for {split}:[/bold green]")
        rprint(training_metrics[split])
    if training_metrics:
        with open(
            os.path.join(results_dir, "evaluation_results.json"),
            "w",
        ) as f:
            json.dump(training_metrics, f, indent=4)

    with torch.inference_mode():
        resource_logger.clear_cuda()
        generate_raw_samples(
            model,
            tokenizer,
            raw_dataset,
            args.gen.gen_post_train_max_samples,
            batch_size=args.gen.gen_batch_size,
            max_new_tokens=args.gen.generate_max_new_tokens,
            do_sample=args.gen.gen_do_sample,
            temperature=args.gen.gen_temperature,
            top_k=args.gen.gen_top_k,
            top_p=args.gen.gen_top_p,
            n_per_sample=args.gen.gen_n_per_sample,
            save_path=os.path.join(results_dir, "fine_tuned"),
        )
        resource_logger.record("generate")

    if (
        args.gen.benchmark_dataset_name_or_path is not None
        and args.gen.benchmark_dataset_type is not None
    ):
        print("Generating raw samples on the benchmark dataset...")
        additional_raw_dataset = load_raw_dataset(
            args.gen.benchmark_dataset_name_or_path,
            dataset_type=args.gen.benchmark_dataset_type,
            # max_train_samples=args.gen.gen_post_train_max_samples,
            # max_validation_samples=args.gen.gen_post_train_max_samples,
            # max_test_samples=args.gen.gen_post_train_max_samples,
            # load_from_cache_file=False,
        )
        task_name = args.gen.benchmark_dataset_name_or_path.split("/")[-1]
        task_name = task_name.split(".")[0]
        with torch.inference_mode():
            resource_logger.clear_cuda()
            generate_raw_samples(
                model,
                tokenizer,
                additional_raw_dataset,
                num_samples=args.gen.gen_post_train_max_samples,
                batch_size=args.gen.gen_batch_size,
                max_new_tokens=args.gen.generate_max_new_tokens,
                do_sample=args.gen.benchmark_do_sample,
                temperature=args.gen.benchmark_temperature,
                top_k=args.gen.benchmark_top_k,
                top_p=args.gen.benchmark_top_p,
                n_per_sample=args.gen.benchmark_n_per_sample,
                save_path=os.path.join(
                    results_dir,
                    f"bench_{task_name}",
                ),
                metadata_field_limit=99999,
            )
            resource_logger.record(f"generate_bench_{task_name}")

    resource_logger.print()


if __name__ == "__main__":
    main()
