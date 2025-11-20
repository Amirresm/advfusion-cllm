from enum import Enum


class DatasetType(Enum):
    CodeSearchNet = "codesearchnet"
    CodeGeneration = "codegeneration"
    CodeTranslation = "ct_dataset"
    CodeTranslationBench = "ct_bench_dataset"
    CMG = "cmg"

    @classmethod
    def get_dataset_type(cls, dataset_name_or_path: str) -> "DatasetType":
        dataset_name_or_path = dataset_name_or_path.lower()
        for category in cls:
            if category.value in dataset_name_or_path:
                return category
        raise ValueError(f"Dataset type not recognized for {dataset_name_or_path}")
