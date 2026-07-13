"""Load POPE samples for generation.

POPE uses the same LLaVA-style JSONL input shape as generic VQA evaluation.
This thin class keeps the adapter registry explicit at the dataset level.
"""

from tinyllava.eval.dataset_adapters.vqa.loader_vqa import VqaLoader


class PopeLoader(VqaLoader):
    pass
