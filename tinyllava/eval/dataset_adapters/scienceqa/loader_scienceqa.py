"""Load ScienceQA samples for generation."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from tinyllava.data.image_payload import resolve_image_payload
from tinyllava.eval.dataset_adapters.loader_base import GenerationExample
from tinyllava.eval.generation import strip_legacy_image_markers


class ScienceQaLoader:
    """Loader for ScienceQA's LLaVA conversion JSON files."""

    def load_samples(self, question_file: str) -> list[Mapping[str, Any]]:
        with open(os.path.expanduser(question_file), encoding="utf-8") as f:
            return json.load(f)

    def make_example(
        self,
        sample: Mapping[str, Any],
        *,
        image_folder: str,
        args: Any,
    ) -> GenerationExample:
        prompt = strip_legacy_image_markers(sample["conversations"][0]["value"])
        if getattr(args, "single_pred_prompt", False):
            prompt += "\nAnswer with the option's letter from the given choices directly."

        image_files = []
        if "image" in sample:
            image_files.append(resolve_image_payload(sample["image"], image_folder=image_folder))

        return GenerationExample(
            question_id=str(sample["id"]),
            prompt=prompt,
            image_files=image_files,
            metadata={"has_image": bool(image_files)},
        )

    def process_response(self, sample: Mapping[str, Any], response: str) -> str:
        return response
