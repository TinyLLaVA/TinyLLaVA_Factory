from collections import OrderedDict


DATASET_LOADER_MAPPING_NAMES = OrderedDict(
    [
        ("vqa", "VqaLoader"),
        ("pope", "PopeLoader"),
        ("scienceqa", "ScienceQaLoader"),
        ("mmmu", "MmmuLoader"),
    ]
)

DATASET_EVALUATION_MAPPING_NAMES = OrderedDict(
    [
        ("vqav2", "Vqav2Evaluation"),
        ("gqa", "GqaEvaluation"),
        ("mmvet", "MmvetEvaluation"),
        ("mmmu", "MmmuEvaluation"),
        ("textvqa", "TextVqaEvaluation"),
        ("pope", "PopeEvaluation"),
        ("scienceqa", "ScienceQaEvaluation"),
    ]
)
