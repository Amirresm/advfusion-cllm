from typing import Literal
import adapters
from adapters.trainer import TrainingArguments
import torch
from transformers.data.data_collator import (
    DefaultDataCollator,
)
from transformers.trainer import Trainer

class DataCollatorWithPaddingAndLabels:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        if self.max_length is not None:
            for f in features:
                if len(f["input_ids"]) > self.max_length:
                    f["input_ids"] = f["input_ids"][: self.max_length]
                if len(f["attention_mask"]) > self.max_length:
                    f["attention_mask"] = f["attention_mask"][: self.max_length]
                if "labels" in f and len(f["labels"]) > self.max_length:
                    f["labels"] = f["labels"][: self.max_length]

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f.get("labels", None) for f in features]

        input_ids = [torch.tensor(ids) for ids in input_ids]
        attention_mask = [torch.tensor(mask) for mask in attention_mask]
        labels = [torch.tensor(lbl) for lbl in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        return batch

def get_trainer(
    model,
    tokenizer,
    peft_lib: Literal["adp", "peft"] | None,
    train_dataset,
    eval_dataset,
    output_dir,
    train_batch_size,
    eval_batch_size,
    epochs,
    learning_rate,
    warmup_ratio,
    optim="paged_adamw_32bit",
    logging_steps=0.05,
    eval_steps=0.1,
    eval_accumulation_steps=50,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    max_length=None,
):
    # data_collator = DefaultDataCollator()
    data_collator = DataCollatorWithPaddingAndLabels(
        tokenizer=tokenizer, max_length=max_length
    )

    # callbacks = []

    bf16 = model.config.torch_dtype == torch.bfloat16
    print(f"Using bf16 for training: {bf16}")

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        optim=optim,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        save_strategy="steps",
        load_best_model_at_end=True,
        bf16=bf16,
        label_names=["labels"],
        gradient_checkpointing=gradient_checkpointing,
    )

    TrainerClass = adapters.AdapterTrainer if peft_lib == "adp" else Trainer
    trainer = TrainerClass(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # callbacks=callbacks,
    )

    return trainer
