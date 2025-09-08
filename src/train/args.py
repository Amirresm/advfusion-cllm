from dataclasses import dataclass


@dataclass
class TrainArgs:
    """Training arguments for the model."""
    do_train: bool = False
    do_eval: bool = False

    train_batch_size: int = 4
    eval_batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    optim: str = "paged_adamw_32bit"
    eval_accumulation_steps: int = 50
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True

    logging_steps: float = 0.05
    eval_steps: float = 0.1
