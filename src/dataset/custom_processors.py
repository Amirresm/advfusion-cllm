from typing import Callable, TypedDict

from src.dataset.utils import DatasetType
import difflib
import inspect


class RawRow(TypedDict):
    TEXT: list[str]
    TARGET: list[str] | None


type RawPreprocessor = Callable[[dict[str, list]], RawRow]


def ct_bench_processor(example: dict) -> RawRow:
    """Code translation dataset processor."""
    prompts = []
    targets = []
    for i in range(len(example["id"])):
        source_lang = example["source_lang"][i]
        target_lang = example["target_lang"][i]

        prompt = example["source_content"][i]
        target = example["target_content"][i]

        prompt = f"### Code written in {source_lang}:\n{prompt}\n### Code written in {target_lang}:\n"
        prompt = prompt + example["prompt"][i]
        prompts.append(prompt)
        targets.append(target)

    return {
        "TEXT": prompts,
        "TARGET": targets,
    }


def ct_processor(example: dict) -> RawRow:
    """Code translation dataset processor."""
    prompts = []
    targets = []
    for i in range(len(example["id"])):
        source_lang = example["source_lang"][i]
        target_lang = example["target_lang"][i]

        prompt = example["source_content"][i]
        target = example["target_content"][i]

        prompt = f"### Code written in {source_lang}:\n{prompt}\n### Code written in {target_lang}:\n"
        prompts.append(prompt)
        targets.append(target)

    return {
        "TEXT": prompts,
        "TARGET": targets,
    }


def csn_processor(example: dict) -> RawRow:
    prompts = []
    targets = []
    for i in range(len(example["code"])):
        prompt = example["code"][i]
        target = (
            example.get("docstring", [""])[i] if "docstring" in example else ""
        )
        if target:
            prompt = prompt.replace(target, "")

        prompt = f"{prompt}\n### Response:\n"
        prompts.append(prompt)
        targets.append(target)

    return {
        "TEXT": prompts,
        "TARGET": targets,
    }


def code_gen_processor(example: dict) -> RawRow:
    prompts = []
    targets = []
    for i in range(len(example["code"])):
        code = example["code"][i]
        s = code.split('"""\n')
        prompt = "".join(s[:-1]) + '"""\n'
        target = s[-1] if len(s) > 1 else ""

        prompts.append(prompt)
        if target:
            targets.append(target)
    return {
        "TEXT": prompts,
        "TARGET": targets,
    }


def cmg_processor(example: dict) -> RawRow:
    prompts = []
    targets = []
    for old_file, new_file, old_content, new_content, subject in zip(
        example["old_file"],
        example["new_file"],
        example["old_contents"],
        example["new_contents"],
        example["subject"],
    ):
        diff = difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=old_file,
            tofile=new_file,
            lineterm="",
        )
        prompt = (
            "Based on the change given, only write commit message.\n"
            + "### Diff:\n"
            + "\n".join(diff)
            + "\n### Commit Message:\n"
        )
        prompts.append(prompt)
        targets.append(subject)

    return {
        "TEXT": prompts,
        "TARGET": targets,
    }


print("Current implementation of cmg_processor:\n")
print(inspect.getsource(cmg_processor))


def get_dataset_processor(dataset_type: DatasetType) -> RawPreprocessor:
    if dataset_type == DatasetType.CodeSearchNet:
        return csn_processor
    elif dataset_type == DatasetType.CodeGeneration:
        return code_gen_processor
    elif dataset_type == DatasetType.CodeTranslation:
        return ct_processor
    elif dataset_type == DatasetType.CMG:
        return cmg_processor
    elif dataset_type == DatasetType.CodeTranslationBench:
        return ct_bench_processor
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
