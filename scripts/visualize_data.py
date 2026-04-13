from dataclasses import dataclass
import os

import simple_parsing
from rich import print as rprint
from torch.utils.data import DataLoader

from src.dataset.args import DatasetArgs
from src.dataset.utils import DatasetType
from src.model.model import init_tokenizer
from src.dataset.dataset import load_raw_dataset, preprocess_dataset
from src.train.trainer import DataCollatorWithPaddingAndLabels
from src.utils.args import parse_args


@dataclass
class Args(simple_parsing.Serializable):
    model_name_or_path: str
    dataset: DatasetArgs

    output_dir: str = (
        ""  # The output directory where the model predictions and checkpoints will be written.
    )


def main():
    args = parse_args(Args)
    assert type(args.dataset.dataset_type) is DatasetType

    tokenizer = init_tokenizer(args.model_name_or_path)

    output_dir = os.path.realpath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_dir = os.path.join(output_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        args.dump_yaml(f)

    raw_dataset = load_raw_dataset(
        args.dataset.dataset_name_or_path,
        dataset_type=args.dataset.dataset_type,
        train_file=args.dataset.train_file,
        test_file=args.dataset.test_file,
        validation_file=args.dataset.validation_file,
        max_train_samples=args.dataset.max_train_samples,
        max_validation_samples=args.dataset.max_eval_samples,
        max_test_samples=args.dataset.max_test_samples,
        load_from_cache_file=False,
    )

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
            debug=True,
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
            debug=True,
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
            debug=True,
        )

        data_collator_cpad = DataCollatorWithPaddingAndLabels(
            tokenizer, max_length=args.dataset.train_max_length
        )

        if train_dataset:
            dataloader = DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=data_collator_cpad,
            )

            print("Visualizing training data")
            for b, batch in enumerate(dataloader):
                rprint("=" * 80 + f"Batch {b + 1}")
                for i in range(len(batch["input_ids"])):
                    input_ids = batch["input_ids"][i]
                    attention_mask = batch["attention_mask"][i]
                    labels = batch["labels"][i]
                    for j in range(len(input_ids)):
                        id = input_ids[j]
                        token = tokenizer.decode(id, skip_special_tokens=False)
                        label = labels[j]
                        color = "green" if label != -100 else "red"
                        rprint(f"[{color}]{token}[/]", end="")
                    print()
                    rprint(
                        len(input_ids),
                        len(attention_mask),
                        len(labels),
                    )
                if b > 2:
                    break


if __name__ == "__main__":
    main()
