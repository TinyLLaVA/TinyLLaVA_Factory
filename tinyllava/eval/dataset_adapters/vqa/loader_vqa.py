"""Load LLaVA-style JSONL VQA samples for generation."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from tinyllava.data.image_payload import resolve_image_payload
from tinyllava.eval.dataset_adapters.loader_base import GenerationExample
from tinyllava.eval.generation import strip_legacy_image_markers


class VqaLoader:
    """Loader for LLaVA-style JSONL VQA files.

    Expected sample shape:
        {"question_id": ..., "text": ..., "image": ...}
    """

    def load_samples(self, question_file: str) -> list[Mapping[str, Any]]:
        with open(os.path.expanduser(question_file), encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def make_example(
        self,
        sample: Mapping[str, Any],
        *,
        image_folder: str,
        args: Any,
    ) -> GenerationExample:
        image_files = [resolve_image_payload(sample["image"], image_folder=image_folder)]
        return GenerationExample(
            question_id=str(sample["question_id"]),
            prompt=strip_legacy_image_markers(str(sample["text"])),
            image_files=image_files,
            metadata={"has_image": True},
        )

    def process_response(self, sample: Mapping[str, Any], response: str) -> str:
        return response
