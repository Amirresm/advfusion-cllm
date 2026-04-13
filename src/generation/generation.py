from collections import defaultdict
import json
import os

import torch
from transformers.modeling_utils import PreTrainedModel
from rich import print as rprint
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.dataset.constants import TARGET_COLUMN, TEXT_COLUMN

from src.evaluate.evaluate import calc_all_metrics
from src.utils.jsonl import write_jsonl


def print_generation_result(
    row, total_count, time_spent, time_remaining, time_remaining2
):
    index = row["index"]
    input = row["input"]
    target = row["target"]
    token_count = row["token_count"]
    generation = row["generation"]
    metrics = row["metrics"]
    rprint(f"[red]{'-' * 20}Sample {index + 1}:[/red]")
    rprint("[cyan]-------Input text:[/cyan]")
    print(input)
    rprint("[cyan]-------Target text:[/cyan]")
    print(target)

    rprint(f"[green]-------Generated text ({token_count} tokens):[/green]")
    print(generation)

    rprint(f"[green]-------Metrics for {index + 1}:[/green]")
    for k, v in metrics.items():
        try:
            rprint(f"{k}: {float(v):.5f}")
        except:
            rprint(f"{k}: {v}")
    time_spent = time_spent or 0
    time_remaining = time_remaining or 0
    time_remaining2 = time_remaining2 or 0

    elapsed_minutes = int(time_spent // 60)
    elapsed_seconds = int(time_spent % 60)
    elapsed_str = f"{elapsed_minutes}m {elapsed_seconds}s"
    remaining_minutes = int(time_remaining // 60)
    remaining_seconds = int(time_remaining % 60)
    remaining_str = f"{remaining_minutes}m {remaining_seconds}s"

    remaining2_minutes = int(time_remaining2 // 60)
    remaining2_seconds = int(time_remaining2 % 60)
    remaining2_str = f"{remaining2_minutes}m {remaining2_seconds}s"

    percentaage_str = f"{(index + 1) / total_count * 100:.2f}%"
    rprint(
        f"[yellow]-------Progress:[/yellow] {index + 1}/{total_count} ({percentaage_str}) "
        f"Elapsed: {elapsed_str}, Remaining: {remaining_str} (alt: {remaining2_str})"
    )


def custom_generation_loop(model, tokenizer, sample_sent):
    tokenized = tokenizer(
        sample_sent,
        return_tensors="pt",
    )
    sample_input = tokenized["input_ids"].to(model.device)
    sample_attn = tokenized["attention_mask"].to(model.device)
    print("Sample sentence:", sample_sent)
    print("Sample input:", sample_input)
    out_tokens = []
    for i in range(10):
        out = model(
            input_ids=sample_input,
            attention_mask=sample_attn,
        )
        logits = out.logits
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        sample_input = torch.cat(
            [sample_input, next_token.unsqueeze(0)],
            dim=1,
        )
        sample_attn = torch.cat(
            [sample_attn, torch.ones((1, 1), device=sample_input.device)],
            dim=1,
        )
        next_token = next_token.item()
        out_tokens.append(next_token)
        token = tokenizer.decode(
            next_token,
            skip_special_tokens=True,
        )
        print(token, end="")

    print("\nGenerated tokens:", out_tokens)


def generate_raw_samples(
    model: PreTrainedModel,
    tokenizer,
    raw_dataset,
    num_samples,
    batch_size,
    max_new_tokens,
    save_path=None,
    metadata_field_limit=100,
    do_sample=False,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    n_per_sample: int = 1,
):
    if num_samples is not None and num_samples <= 0:
        return

    target_split = (
        "test" if "test" in raw_dataset else list(raw_dataset.keys())[0]
    )
    samples = (
        raw_dataset[target_split][:num_samples]
        if num_samples
        else raw_dataset[target_split][:]
    )

    if (
        not samples
        or len(samples[TEXT_COLUMN]) == 0
        or len(samples[TARGET_COLUMN]) == 0
    ):
        print("No samples to generate from.")
        return

    if n_per_sample > 1:
        for k, v in samples.items():
            samples[k] = [item for item in v for _ in range(n_per_sample)]

        samples["duplicate_index"] = [
            i
            for _ in range(len(next(iter(samples.values()))) // n_per_sample)
            for i in range(n_per_sample)
        ]

    input_column = TEXT_COLUMN
    target_column = TARGET_COLUMN
    if model.config.pad_token_id is None:
        print("Setting pad_token_id to eos_token_id")
        model.config.pad_token_id = tokenizer.eos_token_id
    if (
        model.generation_config is not None
        and model.generation_config.pad_token_id is None
    ):
        print("Setting generation_config.pad_token_id to eos_token_id")
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    generations_path = f"{save_path}_samples.jsonl" if save_path else None
    # if generations_path and os.path.exists(generations_path):
    #     os.remove(generations_path)

    initial_index = 0
    if generations_path and os.path.exists(generations_path):
        existing_indices = set()
        with open(generations_path, "r") as f:
            for line in f:
                row = json.loads(line)
                existing_indices.add(row["index"])
        print(
            f"Found {len(existing_indices)} existing samples in {generations_path}, skipping them."
        )
        samples = {
            k: [v[i] for i in range(len(v)) if i not in existing_indices]
            for k, v in samples.items()
        }
        initial_index = len(existing_indices)
        if len(samples[input_column]) == 0:
            print("All samples already generated, skipping.")
            return

    results = []
    batches = []
    for i in range(0, len(samples[input_column]), batch_size):
        batch = []
        for j in range(batch_size):
            if i + j < len(samples[input_column]):
                input = samples[input_column][i + j]
                target = samples[target_column][i + j]
                metadata = {
                    k: (
                        samples[k][i + j]
                        if (
                            not isinstance(samples[k][i + j], str)
                            or len(samples[k][i + j]) < metadata_field_limit
                        )
                        else samples[k][i + j][: metadata_field_limit - 3]
                        + "..."
                    )
                    for k in samples
                    if k not in [input_column, target_column]
                }
                batch.append((input, target, metadata))
        batches.append(batch)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False,
    ) as pbar:
        total_samples = sum([len(b) for b in batches])
        pbar_task = pbar.add_task(
            f"Generating (bs={batch_size})", total=total_samples
        )

        for j, batch in enumerate(batches):
            tokenized = tokenizer(
                [b[0] for b in batch],
                return_tensors="pt",
                padding="longest",
                # truncation=True,
                # max_length=512,
            )
            generations = model.generate(
                input_ids=tokenized["input_ids"].to(model.device),
                attention_mask=tokenized["attention_mask"].to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            prompt_length = len(tokenized["input_ids"][0])
            for i, gen in enumerate(generations):
                index = j * batch_size + i
                # input = batch[i][0]
                input_ids = tokenized["input_ids"][i]
                input = tokenizer.decode(
                    input_ids,
                    skip_special_tokens=True,
                )
                target = batch[i][1]
                metadata = batch[i][2]

                new_tokens = gen
                new_tokens = new_tokens[prompt_length:]
                generation = tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                )
                token_count = len(new_tokens)

                if target:
                    metrics = calc_all_metrics([generation], [target])
                else:
                    metrics = {}

                result_row = {
                    "index": index + initial_index,
                    "input": input,
                    "target": target,
                    "generation": generation,
                    "token_count": token_count,
                    "metrics": metrics,
                }
                results.append(result_row)
                result_row = result_row.copy()

                if metrics:
                    pretty_metrics = {}
                    for k, v in metrics.items():
                        if type(v) is float:
                            pretty_metrics[k] = f"{v:.5f}"
                        if type(v) is list:
                            pretty_metrics[k] = [
                                f"{x:.5f}" if type(x) is float else x for x in v
                            ]
                        else:
                            pretty_metrics[k] = str(v)
                    result_row["metrics"] = pretty_metrics

                print_generation_result(
                    result_row,
                    total_samples,
                    pbar.tasks[0].elapsed,
                    pbar.tasks[0].time_remaining,
                    pbar.tasks[0].remaining,
                )
                result_row["metadata"] = metadata
                if generations_path is not None:
                    write_jsonl(generations_path, [result_row], append=True)
            pbar.update(pbar_task, advance=len(batch))
            pbar.refresh()

    if len(results) == 0:
        return

    total_metrics = calc_all_metrics(
        [r["generation"] for r in results],
        [r["target"] for r in results],
    )

    averages = defaultdict(float)
    for res in results:
        for k, v in res["metrics"].items():
            if type(v) is float:
                averages[k] += v
    for k in averages:
        averages[k] /= len(results)
    averages = dict(averages)

    rprint(f"\n[red]{'Metrics Summary':-^40}[/red]")
    rprint("Total metrics:", total_metrics)
    rprint("Averages:", averages)

    metrics = {
        "total_metrics": total_metrics,
        "averages": averages,
        "metadata": {
            "batch_size": batch_size,
            "num_samples": len(results),
        },
    }

    if save_path is not None:
        file_path = f"{save_path}_metrics.json"
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)

    tokenizer.padding_side = old_pad_side
