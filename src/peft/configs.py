from dataclasses import dataclass
from typing import Literal

import adapters
import peft


@dataclass
class PeftConfigs:
    LoRA = peft.LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=None,
    )


@dataclass
class AdaptersConfigsClass:
    SeqBn = adapters.SeqBnConfig()
    SeqBnInv = adapters.SeqBnInvConfig()
    LoRA = adapters.LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.1,
        selfattn_lora=True,
        intermediate_lora=False,
        output_lora=False,
        attn_matrices=["q", "v"],
    )
    Compacter = adapters.CompacterConfig(
        reduction_factor=32,
        phm_dim=4,
    )


AdaptersConfigs = AdaptersConfigsClass()


def get_peft_config(peft_method: str, lib: Literal["peft", "adp"]):
    if lib == "peft":
        if peft_method == "lora":
            return PeftConfigs.LoRA
        else:
            raise ValueError(f"Unsupported PEFT method: {peft_method}")
    elif lib == "adp":
        if peft_method == "seq_bn":
            return AdaptersConfigs.SeqBn
        elif peft_method == "seq_bn_inv":
            return AdaptersConfigs.SeqBnInv
        elif peft_method == "lora":
            return AdaptersConfigs.LoRA
        elif peft_method == "compacter":
            return AdaptersConfigs.Compacter
        else:
            raise ValueError(f"Unsupported Adapters method: {peft_method}")
    else:
        raise ValueError(f"Unsupported library: {lib}")
